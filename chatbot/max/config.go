package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/caarlos0/env/v10"
)

// Config aggregates runtime configuration loaded from environment variables.
type Config struct {
	MaxToken        string        `env:"MAX_ACCESS_TOKEN,required"`
	MaxAPIBase      string        `env:"MAX_API_BASE"`
	MaxAPIVersion   string        `env:"MAX_API_VERSION"`
	MaxAPITimeout   time.Duration `env:"MAX_API_TIMEOUT"`
	MaxDebug        bool          `env:"MAX_API_DEBUG"`
	MaxDebugChat    int64         `env:"MAX_DEBUG_CHAT"`
	QAHost          string        `env:"QA_HOST"`
	DatabaseURL     string        `env:"DATABASE_URL"`
	PostgresUser    string        `env:"POSTGRES_USER"`
	PostgresPass    string        `env:"POSTGRES_PASSWORD"`
	PostgresHost    string        `env:"POSTGRES_HOST"`
	PostgresDB      string        `env:"POSTGRES_DB"`
	ConfluenceHost  string        `env:"CONFLUENCE_HOST"`
	ConfluenceToken string        `env:"CONFLUENCE_TOKEN"`
	SpacesRaw       string        `env:"CONFLUENCE_SPACES"`
}

func loadConfig() (Config, error) {
	cfg := Config{}
	if err := env.Parse(&cfg); err != nil {
		return Config{}, err
	}

	if cfg.MaxAPIBase == "" {
		cfg.MaxAPIBase = "https://platform-api.max.ru/"
	}
	if cfg.MaxAPIVersion == "" {
		cfg.MaxAPIVersion = "1.2.5"
	}
	if cfg.MaxAPITimeout == 0 {
		cfg.MaxAPITimeout = 30 * time.Second
	}
	if cfg.QAHost == "" {
		cfg.QAHost = "qa:8080"
	}
	if cfg.DatabaseURL == "" {
		if cfg.PostgresUser == "" || cfg.PostgresPass == "" || cfg.PostgresHost == "" || cfg.PostgresDB == "" {
			return Config{}, fmt.Errorf("database configuration is incomplete")
		}
		cfg.DatabaseURL = fmt.Sprintf(
			"postgresql://%s:%s@%s/%s?sslmode=disable",
			cfg.PostgresUser,
			cfg.PostgresPass,
			cfg.PostgresHost,
			cfg.PostgresDB,
		)
	}
	cfg.SpacesRaw = strings.TrimSpace(cfg.SpacesRaw)
	return cfg, nil
}

// ConfluenceSpaces returns parsed space keys from the raw env variable.
func (c Config) ConfluenceSpaces() []string {
	if c.SpacesRaw == "" {
		return nil
	}
	return strings.Fields(c.SpacesRaw)
}
