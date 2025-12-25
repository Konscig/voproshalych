package main

import "github.com/max-messenger/max-bot-api-client-go/configservice"

// maxAPIConfig implements configservice.ConfigInterface using runtime settings.
type maxAPIConfig struct {
	baseURL   string
	timeout   int
	version   string
	token     string
	debug     bool
	debugChat int64
}

var _ configservice.ConfigInterface = (*maxAPIConfig)(nil)

func (m *maxAPIConfig) GetHttpBotAPIUrl() string {
	return m.baseURL
}

func (m *maxAPIConfig) GetHttpBotAPITimeOut() int {
	return m.timeout
}

func (m *maxAPIConfig) GetHttpBotAPIVersion() string {
	return m.version
}

func (m *maxAPIConfig) BotTokenCheckInInputSteam() bool {
	return false
}

func (m *maxAPIConfig) BotTokenCheckString() string {
	return m.token
}

func (m *maxAPIConfig) GetDebugLogMode() bool {
	return m.debug
}

func (m *maxAPIConfig) GetDebugLogChat() int64 {
	return m.debugChat
}
