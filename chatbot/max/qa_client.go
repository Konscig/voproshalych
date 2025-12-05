package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	qaPathQuestion    = "/qa/"
	qaPathAnswerEmbed = "/answer_embed/"
)

// QAClient wraps access to the Python QA microservice.
type QAClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewQAClient(host string) *QAClient {
	base := normalizeServiceHost(host)
	return &QAClient{
		baseURL: base,
		httpClient: &http.Client{
			Timeout: 20 * time.Second,
		},
	}
}

func (c *QAClient) Ask(ctx context.Context, question string, dialogContext []string) (string, string, error) {
	payload := map[string]interface{}{
		"question":       strings.TrimSpace(strings.ToLower(question)),
		"dialog_context": dialogContext,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+qaPathQuestion, bytes.NewReader(body))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("qa service returned %d", resp.StatusCode)
	}

	var result struct {
		Answer        string `json:"answer"`
		ConfluenceURL string `json:"confluence_url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", err
	}
	return result.Answer, result.ConfluenceURL, nil
}

func (c *QAClient) ReportGoodAnswer(ctx context.Context, qaID int64) error {
	payload := map[string]int64{"answer_id": qaID}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+qaPathAnswerEmbed, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func normalizeServiceHost(host string) string {
	h := strings.TrimSpace(host)
	if h == "" {
		return "http://localhost"
	}
	if strings.HasPrefix(h, "http://") || strings.HasPrefix(h, "https://") {
		return strings.TrimRight(h, "/")
	}
	return "http://" + strings.TrimRight(h, "/")
}
