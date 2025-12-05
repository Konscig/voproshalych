package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/joho/godotenv"
	maxbot "github.com/max-messenger/max-bot-api-client-go"

	"github.com/Bolshevichok/vopromax/chatbot/storage"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("bot stopped: %v", err)
	}
}

func run() error {
	// Load .env for local development, ignore errors if the file does not exist.
	_ = godotenv.Load("../.env")
	_ = godotenv.Load()

	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	apiCfg := &maxAPIConfig{
		baseURL:   ensureTrailingSlash(cfg.MaxAPIBase),
		timeout:   int(cfg.MaxAPITimeout.Seconds()),
		version:   cfg.MaxAPIVersion,
		token:     cfg.MaxToken,
		debug:     cfg.MaxDebug,
		debugChat: cfg.MaxDebugChat,
	}
	api, err := maxbot.NewWithConfig(apiCfg)
	if err != nil {
		return err
	}

	store, err := storage.New(cfg.DatabaseURL)
	if err != nil {
		return err
	}
	qaClient := NewQAClient(cfg.QAHost)
	confClient := NewConfluenceClient(cfg)
	bot := NewBot(api, store, qaClient, confClient)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		signals := make(chan os.Signal, 1)
		signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
		<-signals
		cancel()
	}()

	log.Printf("Max bot started. QA host=%s", cfg.QAHost)

	for upd := range api.GetUpdates(ctx) {
		bot.HandleUpdate(ctx, upd)
	}

	return ctx.Err()
}

func ensureTrailingSlash(base string) string {
	trimmed := strings.TrimSpace(base)
	if trimmed == "" {
		return "https://platform-api.max.ru/"
	}
	if !strings.HasSuffix(trimmed, "/") {
		trimmed += "/"
	}
	return trimmed
}
