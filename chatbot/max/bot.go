package main

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	maxbot "github.com/max-messenger/max-bot-api-client-go"
	"github.com/max-messenger/max-bot-api-client-go/schemes"

	"github.com/Konscig/voproshalych/chatbot/max/storage"
)

// Bot wires external services and reacts to incoming MAX updates.
type Bot struct {
	api   *maxbot.Api
	store *storage.Store
	qa    *QAClient
	conf  *ConfluenceClient
}

func NewBot(api *maxbot.Api, store *storage.Store, qa *QAClient, conf *ConfluenceClient) *Bot {
	return &Bot{api: api, store: store, qa: qa, conf: conf}
}

func (b *Bot) HandleUpdate(ctx context.Context, upd schemes.UpdateInterface) {
	switch u := upd.(type) {
	case *schemes.MessageCreatedUpdate:
		text := strings.TrimSpace(u.GetText())
		if text == "" {
			return
		}
		if err := b.handleText(ctx, u.Message.Sender.UserId, text); err != nil {
			log.Printf("handleText error: %v", err)
		}
	case *schemes.MessageCallbackUpdate:
		payload := strings.TrimSpace(u.Callback.Payload)
		if payload == "" {
			return
		}
		if err := b.handlePayload(ctx, u.Callback.User.UserId, payload); err != nil {
			log.Printf("handlePayload error: %v", err)
		}
	case *schemes.BotStartedUpdate:
		if err := b.handleBotStarted(ctx, u.User.UserId); err != nil {
			log.Printf("handleBotStarted error: %v", err)
		}
	default:
		log.Printf("Unhandled update type: %s", upd.GetUpdateType())
	}
}

func (b *Bot) handleBotStarted(ctx context.Context, userID int64) error {
	if _, _, err := b.store.EnsureUser(ctx, userID); err != nil {
		return err
	}
	return b.sendStart(ctx, userID)
}

func (b *Bot) handlePayload(ctx context.Context, userID int64, payload string) error {
	switch {
	case strings.HasPrefix(payload, payloadConfPrefix):
		return b.handleConfluence(ctx, userID, strings.TrimPrefix(payload, payloadConfPrefix))
	case strings.HasPrefix(payload, payloadRatePrefix):
		return b.handleRatePayload(ctx, userID, payload)
	case strings.HasPrefix(payload, payloadTextPrefix):
		return b.handleText(ctx, userID, strings.TrimPrefix(payload, payloadTextPrefix))
	default:
		return b.handleText(ctx, userID, payload)
	}
}

func (b *Bot) handleText(ctx context.Context, userID int64, text string) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	lower := strings.ToLower(text)
	_, dbID, err := b.store.EnsureUser(ctx, userID)
	if err != nil {
		return err
	}
	switch {
	case lower == BotStrings.Start || lower == BotStrings.StartEnglish:
		return b.sendStart(ctx, userID)
	case lower == strings.ToLower(BotStrings.ConfluenceButton):
		return b.handleHelp(ctx, userID)
	case lower == strings.ToLower(BotStrings.Subscribe) || lower == strings.ToLower(BotStrings.Unsubscribe):
		return b.toggleSubscription(ctx, dbID, userID)
	case lower == strings.ToLower(BotStrings.NewDialog):
		return b.resetDialog(ctx, dbID, userID)
	case strings.HasPrefix(lower, "conf_id") || isDigits(lower):
		return b.handleConfluenceText(ctx, userID, lower)
	case strings.HasPrefix(lower, "rate"):
		return b.handleManualRate(ctx, userID, lower)
	}
	if len([]rune(text)) < 4 {
		return b.sendWithKeyboard(ctx, userID, BotStrings.Less4Symbols, helpKeyboardLayout())
	}
	spam, err := b.store.CheckSpam(ctx, dbID)
	if err != nil {
		return err
	}
	if spam {
		return b.sendWithKeyboard(ctx, userID, BotStrings.SpamWarning, helpKeyboardLayout())
	}
	return b.answerQuestion(ctx, dbID, userID, text)
}

func (b *Bot) sendStart(ctx context.Context, userID int64) error {
	return b.sendWithKeyboard(ctx, userID, BotStrings.FirstMessage, helpKeyboardLayout())
}

func (b *Bot) toggleSubscription(ctx context.Context, userID int64, maxID int64) error {
	isSubscribed, err := b.store.ToggleSubscription(ctx, userID)
	if err != nil {
		return err
	}
	msg := BotStrings.UnsubscribeMessage
	if isSubscribed {
		msg = BotStrings.SubscribeMessage
	}
	return b.sendWithKeyboard(ctx, maxID, msg, helpKeyboardLayout())
}

func (b *Bot) resetDialog(ctx context.Context, userID int64, maxID int64) error {
	if err := b.store.SetStopPoint(ctx, userID, true); err != nil {
		return err
	}
	return b.sendWithKeyboard(ctx, maxID, BotStrings.DialogReset, helpKeyboardLayout())
}

func (b *Bot) handleHelp(ctx context.Context, userID int64) error {
	return b.sendWithKeyboard(ctx, userID, BotStrings.HelpVPNNotice, helpKeyboardLayout())
}

func (b *Bot) handleConfluenceText(ctx context.Context, userID int64, lowered string) error {
	id := strings.NewReplacer("conf_id", "", ":", "", " ", "").Replace(lowered)
	id = strings.TrimSpace(id)
	if id == "" {
		return nil
	}
	return b.handleConfluence(ctx, userID, id)
}

func (b *Bot) handleConfluence(ctx context.Context, userID int64, pageID string) error {
	if b.conf == nil {
		return b.sendWithKeyboard(ctx, userID, BotStrings.NotAvailable, helpKeyboardLayout())
	}
	res, err := b.conf.ParseByID(ctx, pageID)
	if err != nil {
		return b.sendWithKeyboard(ctx, userID, BotStrings.NotAvailable, helpKeyboardLayout())
	}
	if len(res.Entries) > 0 {
		return b.sendWithKeyboard(ctx, userID, BotStrings.WhichInfo, confluenceKeyboardLayout(res.Entries))
	}
	text := res.Text
	if strings.TrimSpace(text) == "" {
		text = BotStrings.NotAvailable
	}
	return b.sendWithKeyboard(ctx, userID, text, nil)
}

func (b *Bot) handleRatePayload(ctx context.Context, userID int64, payload string) error {
	data := strings.Split(payload, ":")
	if len(data) < 3 {
		return nil
	}
	score, err := strconv.Atoi(data[1])
	if err != nil {
		return nil
	}
	qaID, err := strconv.ParseInt(data[2], 10, 64)
	if err != nil {
		return nil
	}
	return b.processRating(ctx, userID, qaID, score)
}

func (b *Bot) handleManualRate(ctx context.Context, userID int64, text string) error {
	parts := strings.Fields(text)
	if len(parts) < 3 {
		return nil
	}
	score, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil
	}
	qaID, err := strconv.ParseInt(parts[2], 10, 64)
	if err != nil {
		return nil
	}
	return b.processRating(ctx, userID, qaID, score)
}

func (b *Bot) processRating(ctx context.Context, userID int64, qaID int64, score int) error {
	if score == 5 {
		if err := b.qa.ReportGoodAnswer(ctx, qaID); err != nil {
			log.Printf("report good answer failed: %v", err)
		}
	}
	saved, err := b.store.RateAnswer(ctx, qaID, score)
	if err != nil || !saved {
		return err
	}
	return b.sendMessage(ctx, userID, BotStrings.ThanksForFeedback, nil)
}

func (b *Bot) answerQuestion(ctx context.Context, dbUserID int64, maxUserID int64, question string) error {
	if err := b.sendMessage(ctx, maxUserID, BotStrings.TryFindAnswer, nil); err != nil {
		return err
	}
	history, err := b.store.GetHistory(ctx, dbUserID, 30, 5)
	if err != nil {
		return err
	}
	answered, unanswered := storage.FilterHistory(history)
	dialogContext := buildDialogContext(answered, unanswered)
	answer, url, err := b.qa.Ask(ctx, question, dialogContext)
	if err != nil {
		log.Printf("QA request failed: %v", err)
	}
	var urlPtr *string
	url = strings.TrimSpace(url)
	if url != "" {
		urlPtr = &url
	}
	qaID, err := b.store.AddQuestionAnswer(ctx, dbUserID, question, answer, urlPtr)
	if err != nil {
		return err
	}
	if urlPtr == nil {
		return b.sendWithKeyboard(ctx, maxUserID, BotStrings.NotFound, helpKeyboardLayout())
	}
	if strings.TrimSpace(answer) == "" {
		answer = BotStrings.NotAnswer
	}
	msg := fmt.Sprintf("%s\n\n%s %s\n\nОцените ответ: ❤ или 👎", answer, BotStrings.SourceURL, url)
	return b.sendWithKeyboard(ctx, maxUserID, msg, ratingKeyboardLayout(qaID))
}

func (b *Bot) sendWithKeyboard(ctx context.Context, userID int64, text string, builder keyboardBuilder) error {
	return b.sendMessage(ctx, userID, text, builder)
}

func (b *Bot) sendMessage(ctx context.Context, userID int64, text string, builder keyboardBuilder) error {
	msg := maxbot.NewMessage().SetUser(userID).SetText(text)
	if builder != nil {
		keyboard := b.api.Messages.NewKeyboardBuilder()
		builder(keyboard)
		msg.AddKeyboard(keyboard)
	}
	_, err := b.api.Messages.Send(ctx, msg)
	if apiErr, ok := err.(*schemes.Error); ok && apiErr.Code == "" {
		return nil
	}
	return err
}

func buildDialogContext(answered, unanswered []storage.QuestionAnswer) []string {
	context := make([]string, 0, len(answered)*2+len(unanswered))
	for _, qa := range answered {
		context = append(context, fmt.Sprintf("Q: %s", qa.Question))
		context = append(context, fmt.Sprintf("A: %s", qa.Answer))
	}
	for _, qa := range unanswered {
		context = append(context, fmt.Sprintf("Q: %s", qa.Question))
	}
	return context
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
