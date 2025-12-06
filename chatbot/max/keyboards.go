package main

import (
	"fmt"

	maxbot "github.com/max-messenger/max-bot-api-client-go"
	"github.com/max-messenger/max-bot-api-client-go/schemes"
)

const (
	payloadTextPrefix = "text:"
	payloadConfPrefix = "conf_id:"
	payloadRatePrefix = "rate:"
)

// keyboardBuilder mutates the provided keyboard builder in-place.
type keyboardBuilder func(*maxbot.Keyboard)

func helpKeyboardLayout() keyboardBuilder {
	return func(kb *maxbot.Keyboard) {
		kb.AddRow().AddCallback(BotStrings.ConfluenceButton, schemes.DEFAULT, payloadTextPrefix+BotStrings.ConfluenceButton)
		kb.AddRow().AddCallback(BotStrings.NewDialog, schemes.DEFAULT, payloadTextPrefix+BotStrings.NewDialog)
	}
}

func confluenceKeyboardLayout(entries []ConfluenceEntry) keyboardBuilder {
	return func(kb *maxbot.Keyboard) {
		for _, entry := range entries {
			row := kb.AddRow()
			row.AddCallback(entry.Title, schemes.DEFAULT, payloadConfPrefix+entry.ID)
		}
	}
}

func ratingKeyboardLayout(qaID int64) keyboardBuilder {
	return func(kb *maxbot.Keyboard) {
		row := kb.AddRow()
		row.AddCallback("👎", schemes.NEGATIVE, fmt.Sprintf("%s1:%d", payloadRatePrefix, qaID))
		row.AddCallback("❤", schemes.POSITIVE, fmt.Sprintf("%s5:%d", payloadRatePrefix, qaID))
	}
}
