package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
)

const (
	confluenceLabel = "справка"
	cacheTTL        = time.Hour
)

// ConfluenceEntry represents a single selectable page in the inline keyboard.
type ConfluenceEntry struct {
	ID    string
	Title string
}

// ConfluenceResult encapsulates either nested entries or rendered page text.
type ConfluenceResult struct {
	Entries []ConfluenceEntry
	Text    string
}

// ConfluenceClient performs limited read-only operations required by the bot.
type ConfluenceClient struct {
	baseURL string
	token   string
	space   string
	client  *http.Client

	cacheMu    sync.RWMutex
	markup     cachedEntries
	pagesCache map[string]cachedPage
}

type cachedEntries struct {
	data    []ConfluenceEntry
	expires time.Time
}

type cachedPage struct {
	result  ConfluenceResult
	expires time.Time
}

// NewConfluenceClient builds a client if all configuration fields are present.
func NewConfluenceClient(cfg Config) *ConfluenceClient {
	spaces := cfg.ConfluenceSpaces()
	if cfg.ConfluenceHost == "" || cfg.ConfluenceToken == "" || len(spaces) == 0 {
		return nil
	}
	return &ConfluenceClient{
		baseURL: strings.TrimRight(cfg.ConfluenceHost, "/"),
		token:   cfg.ConfluenceToken,
		space:   spaces[0],
		client: &http.Client{
			Timeout: 15 * time.Second,
		},
		pagesCache: make(map[string]cachedPage),
	}
}

// RootEntries returns cached or fresh list of top-level help sections.
func (c *ConfluenceClient) RootEntries(ctx context.Context) ([]ConfluenceEntry, error) {
	if c == nil {
		return nil, errors.New("confluence disabled")
	}
	if entries, ok := c.getCachedRoot(); ok {
		return entries, nil
	}
	homeID, err := c.fetchHomepageID(ctx)
	if err != nil {
		return nil, err
	}
	entries, err := c.fetchChildren(ctx, homeID)
	if err != nil {
		return nil, err
	}
	c.setCachedRoot(entries)
	return entries, nil
}

// ParseByID returns nested entries if the page has children, otherwise renders text content.
func (c *ConfluenceClient) ParseByID(ctx context.Context, id string) (ConfluenceResult, error) {
	if c == nil {
		return ConfluenceResult{}, errors.New("confluence disabled")
	}
	if res, ok := c.getCachedPage(id); ok {
		return res, nil
	}
	entries, err := c.fetchChildren(ctx, id)
	if err == nil && len(entries) > 0 {
		result := ConfluenceResult{Entries: entries}
		c.setCachedPage(id, result)
		return result, nil
	}
	text, err := c.fetchPageText(ctx, id)
	if err != nil {
		return ConfluenceResult{}, err
	}
	result := ConfluenceResult{Text: text}
	c.setCachedPage(id, result)
	return result, nil
}

func (c *ConfluenceClient) fetchHomepageID(ctx context.Context) (string, error) {
	endpoint := fmt.Sprintf("/rest/api/space/%s", url.PathEscape(c.space))
	params := url.Values{}
	params.Set("expand", "homepage")
	var resp struct {
		Homepage struct {
			ID string `json:"id"`
		} `json:"homepage"`
	}
	if err := c.getJSON(ctx, endpoint, params, &resp); err != nil {
		return "", err
	}
	if resp.Homepage.ID == "" {
		return "", fmt.Errorf("homepage id not found for space %s", c.space)
	}
	return resp.Homepage.ID, nil
}

func (c *ConfluenceClient) fetchChildren(ctx context.Context, parentID string) ([]ConfluenceEntry, error) {
	endpoint := "/rest/api/content/search"
	params := url.Values{}
	params.Set("cql", fmt.Sprintf("parent=%s and label=\"%s\"", parentID, confluenceLabel))
	params.Set("limit", "100")
	var resp struct {
		Results []struct {
			Content struct {
				ID    string `json:"id"`
				Title string `json:"title"`
			} `json:"content"`
		} `json:"results"`
	}
	if err := c.getJSON(ctx, endpoint, params, &resp); err != nil {
		return nil, err
	}
	entries := make([]ConfluenceEntry, 0, len(resp.Results))
	for _, r := range resp.Results {
		entry := ConfluenceEntry{ID: r.Content.ID, Title: r.Content.Title}
		if entry.ID != "" && entry.Title != "" {
			entries = append(entries, entry)
		}
	}
	return entries, nil
}

func (c *ConfluenceClient) fetchPageText(ctx context.Context, pageID string) (string, error) {
	endpoint := path.Join("/rest/api/content", pageID)
	params := url.Values{}
	params.Set("expand", "body.storage,_links")
	var resp struct {
		Body struct {
			Storage struct {
				Value string `json:"value"`
			} `json:"storage"`
		} `json:"body"`
		Links struct {
			Base  string `json:"base"`
			WebUI string `json:"webui"`
		} `json:"_links"`
	}
	if err := c.getJSON(ctx, endpoint, params, &resp); err != nil {
		return "", err
	}
	link := strings.TrimRight(resp.Links.Base, "/") + resp.Links.WebUI
	text, err := renderHTMLToText(resp.Body.Storage.Value, link)
	if err != nil {
		return "", err
	}
	return text, nil
}

func (c *ConfluenceClient) getJSON(ctx context.Context, endpoint string, params url.Values, target interface{}) error {
	resp, err := c.request(ctx, endpoint, params)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("confluence request failed: %s", resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(target)
}

func (c *ConfluenceClient) request(ctx context.Context, endpoint string, params url.Values) (*http.Response, error) {
	u := c.baseURL + endpoint
	if len(params) > 0 {
		u += "?" + params.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Accept", "application/json")
	return c.client.Do(req)
}

func (c *ConfluenceClient) getCachedRoot() ([]ConfluenceEntry, bool) {
	c.cacheMu.RLock()
	defer c.cacheMu.RUnlock()
	if time.Now().Before(c.markup.expires) && len(c.markup.data) > 0 {
		copyData := append([]ConfluenceEntry(nil), c.markup.data...)
		return copyData, true
	}
	return nil, false
}

func (c *ConfluenceClient) setCachedRoot(entries []ConfluenceEntry) {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()
	c.markup = cachedEntries{
		data:    append([]ConfluenceEntry(nil), entries...),
		expires: time.Now().Add(cacheTTL),
	}
}

func (c *ConfluenceClient) getCachedPage(id string) (ConfluenceResult, bool) {
	c.cacheMu.RLock()
	defer c.cacheMu.RUnlock()
	item, ok := c.pagesCache[id]
	if !ok || time.Now().After(item.expires) {
		return ConfluenceResult{}, false
	}
	return item.result, true
}

func (c *ConfluenceClient) setCachedPage(id string, result ConfluenceResult) {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()
	c.pagesCache[id] = cachedPage{
		result:  result,
		expires: time.Now().Add(cacheTTL),
	}
}

func renderHTMLToText(htmlBody, link string) (string, error) {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(htmlBody))
	if err != nil {
		return "", err
	}
	doc.Find("strong").Each(func(_ int, sel *goquery.Selection) {
		sel.ReplaceWithSelection(sel.Contents())
	})
	doc.Find("br").Each(func(_ int, sel *goquery.Selection) {
		sel.ReplaceWithHtml("\n")
	})
	doc.Find("ac\\:parameter").Each(func(_ int, sel *goquery.Selection) {
		sel.Remove()
	})
	var builder strings.Builder
	doc.Find("p, li").Each(func(_ int, sel *goquery.Selection) {
		text := strings.TrimSpace(sel.Text())
		if text == "" {
			return
		}
		if goquery.NodeName(sel) == "li" {
			builder.WriteString("• ")
		}
		builder.WriteString(text)
		builder.WriteString("\n\n")
	})
	result := strings.TrimSpace(builder.String())
	if result == "" {
		return fmt.Sprintf("Информация находится по ссылке: %s", link), nil
	}
	return result, nil
}
