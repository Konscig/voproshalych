package storage

import (
	"context"
	"errors"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// User mirrors the legacy SQLAlchemy model.
type User struct {
	ID             int64            `gorm:"primaryKey"`
	MaxID          int64            `gorm:"uniqueIndex"`
	IsSubscribed   bool             `gorm:"not null"`
	QuestionAnswer []QuestionAnswer `gorm:"foreignKey:UserID"`
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

// TableName enforces the legacy table naming convention.
func (User) TableName() string {
	return "user"
}

// QuestionAnswer stores user questions along with answers.
type QuestionAnswer struct {
	ID            int64  `gorm:"primaryKey"`
	Question      string `gorm:"type:text;not null"`
	Answer        string `gorm:"type:text"`
	ConfluenceURL string `gorm:"type:text"`
	Score         *int
	UserID        int64 `gorm:"index"`
	StopPoint     bool  `gorm:"not null;default:false"`
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// TableName enforces the legacy table naming convention.
func (QuestionAnswer) TableName() string {
	return "question_answer"
}

// Store wraps a GORM connection and exposes helper methods.
type Store struct {
	db *gorm.DB
}

// New initialises the database connection and runs migrations.
func New(dsn string) (*Store, error) {
	cfg := &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)}
	db, err := gorm.Open(postgres.Open(dsn), cfg)
	if err != nil {
		return nil, err
	}
	if err := db.AutoMigrate(&User{}, &QuestionAnswer{}); err != nil {
		return nil, err
	}
	return &Store{db: db}, nil
}

// NewWithDB creates a Store using an already opened gorm.DB (for tests)
func NewWithDB(db *gorm.DB) *Store {
	return &Store{db: db}
}

// EnsureUser creates a record for the given MAX id if it doesn't exist.
func (s *Store) EnsureUser(ctx context.Context, maxID int64) (bool, int64, error) {
	var user User
	tx := s.db.WithContext(ctx).Where("max_id = ?", maxID).First(&user)
	if errors.Is(tx.Error, gorm.ErrRecordNotFound) {
		user = User{MaxID: maxID, IsSubscribed: true}
		if err := s.db.WithContext(ctx).Create(&user).Error; err != nil {
			return false, 0, err
		}
		return true, user.ID, nil
	}
	if tx.Error != nil {
		return false, 0, tx.Error
	}
	return false, user.ID, nil
}

// ToggleSubscription switches subscription flag and returns the new state.
func (s *Store) ToggleSubscription(ctx context.Context, userID int64) (bool, error) {
	var user User
	if err := s.db.WithContext(ctx).First(&user, userID).Error; err != nil {
		return false, err
	}
	user.IsSubscribed = !user.IsSubscribed
	if err := s.db.WithContext(ctx).Save(&user).Error; err != nil {
		return false, err
	}
	return user.IsSubscribed, nil
}

// CheckSpam replicates the legacy "more than 5 questions within a minute" heuristic.
func (s *Store) CheckSpam(ctx context.Context, userID int64) (bool, error) {
	var qas []QuestionAnswer
	if err := s.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at desc").
		Limit(5).
		Find(&qas).Error; err != nil {
		return false, err
	}
	if len(qas) < 5 {
		return false, nil
	}
	fifth := qas[len(qas)-1]
	return time.Since(fifth.CreatedAt) < time.Minute, nil
}

// AddQuestionAnswer inserts a new record and returns its id.
func (s *Store) AddQuestionAnswer(ctx context.Context, userID int64, question, answer string, url *string) (int64, error) {
	confluenceURL := ""
	if url != nil {
		confluenceURL = *url
	}
	qa := QuestionAnswer{
		Question:      question,
		Answer:        answer,
		ConfluenceURL: confluenceURL,
		UserID:        userID,
	}
	if err := s.db.WithContext(ctx).Create(&qa).Error; err != nil {
		return 0, err
	}
	return qa.ID, nil
}

// RateAnswer persists user's score if the record exists.
func (s *Store) RateAnswer(ctx context.Context, qaID int64, score int) (bool, error) {
	updates := map[string]interface{}{"score": score}
	tx := s.db.WithContext(ctx).Model(&QuestionAnswer{}).Where("id = ?", qaID).Updates(updates)
	return tx.RowsAffected > 0, tx.Error
}

// GetHistory returns recent question/answers for dialog context.
func (s *Store) GetHistory(ctx context.Context, userID int64, minutes int, limitPairs int) ([]QuestionAnswer, error) {
	if minutes <= 0 {
		minutes = 30
	}
	cutoff := time.Now().Add(-time.Duration(minutes) * time.Minute)
	var records []QuestionAnswer
	if err := s.db.WithContext(ctx).
		Where("user_id = ? AND created_at >= ?", userID, cutoff).
		Order("created_at asc").
		Find(&records).Error; err != nil {
		return nil, err
	}
	if limitPairs <= 0 {
		limitPairs = 5
	}
	answered := make([]QuestionAnswer, 0)
	for _, qa := range records {
		if qa.Answer != "" {
			answered = append(answered, qa)
		}
	}
	if len(answered) > limitPairs {
		answered = answered[len(answered)-limitPairs:]
	}
	unanswered := make([]QuestionAnswer, 0)
	for _, qa := range records {
		if qa.Answer == "" {
			unanswered = append(unanswered, qa)
		}
	}
	return append(answered, unanswered...), nil
}

// FilterHistory trims history using stop points and ordering rules.
func FilterHistory(history []QuestionAnswer) (answered []QuestionAnswer, unanswered []QuestionAnswer) {
	if len(history) == 0 {
		return nil, nil
	}
	lastStop := -1
	for i, qa := range history {
		if qa.StopPoint {
			lastStop = i
		}
	}
	trimmed := history
	if lastStop >= 0 && lastStop < len(history)-1 {
		trimmed = history[lastStop+1:]
	}
	for _, qa := range trimmed {
		if qa.Answer != "" {
			answered = append(answered, qa)
		} else {
			unanswered = append(unanswered, qa)
		}
	}
	if len(answered) == 0 {
		return nil, unanswered
	}
	lastAnswerTime := answered[len(answered)-1].CreatedAt
	filteredUnanswered := make([]QuestionAnswer, 0, len(unanswered))
	for _, qa := range unanswered {
		if qa.CreatedAt.After(lastAnswerTime) {
			filteredUnanswered = append(filteredUnanswered, qa)
		}
	}
	return answered, filteredUnanswered
}

// SetStopPoint marks the latest QA entry as stop_point.
func (s *Store) SetStopPoint(ctx context.Context, userID int64, val bool) error {
	var qa QuestionAnswer
	tx := s.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("created_at desc").
		First(&qa)
	if errors.Is(tx.Error, gorm.ErrRecordNotFound) {
		return nil
	}
	if tx.Error != nil {
		return tx.Error
	}
	qa.StopPoint = val
	return s.db.WithContext(ctx).Save(&qa).Error
}
