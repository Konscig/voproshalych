package storage_test

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/Konscig/voproshalych/chatbot/max/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

// helper: создаёт in-memory БД и Store через NewWithDB
func newTestStore(t *testing.T) *storage.Store {
	t.Helper()

	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	require.NoError(t, err)

	require.NoError(t, db.AutoMigrate(&storage.User{}, &storage.QuestionAnswer{}))

	return storage.NewWithDB(db)
}

// -------------------------------
// TEST: EnsureUser
// -------------------------------
func TestEnsureUser(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	created, id1, err := s.EnsureUser(ctx, 100)
	require.NoError(t, err)
	assert.True(t, created)
	assert.NotZero(t, id1)

	created, id2, err := s.EnsureUser(ctx, 100)
	require.NoError(t, err)
	assert.False(t, created)
	assert.Equal(t, id1, id2)
}

// -------------------------------
// TEST: ToggleSubscription
// -------------------------------
func TestToggleSubscription(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	_, userID, _ := s.EnsureUser(ctx, 777)

	sub1, err := s.ToggleSubscription(ctx, userID)
	require.NoError(t, err)
	assert.False(t, sub1)

	sub2, err := s.ToggleSubscription(ctx, userID)
	require.NoError(t, err)
	assert.True(t, sub2)
}

// -------------------------------
// TEST: AddQuestionAnswer & RateAnswer
// -------------------------------
func TestAddAndRateAnswer(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	_, userID, _ := s.EnsureUser(ctx, 1)

	url := "https://test.com"
	id, err := s.AddQuestionAnswer(ctx, userID, "Q?", "A!", &url)
	require.NoError(t, err)
	assert.NotZero(t, id)

	ok, err := s.RateAnswer(ctx, id, 5)
	require.NoError(t, err)
	assert.True(t, ok)
}

// -------------------------------
// TEST: CheckSpam
// -------------------------------
func TestCheckSpam(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	_, userID, _ := s.EnsureUser(ctx, 50)

	// 5 сообщений → спам
	for i := 0; i < 5; i++ {
		_, err := s.AddQuestionAnswer(ctx, userID, "q", "a", nil)
		require.NoError(t, err)
	}

	isSpam, err := s.CheckSpam(ctx, userID)
	require.NoError(t, err)
	assert.True(t, isSpam)

	// старим записи, чтобы спам исчез
	time.Sleep(time.Millisecond * 5)

	// БЕЗ прямого доступа к s.db:
	// Создадим fresh store поверх той же DB
	db := getInternalDB(t, s)
	db.Model(&storage.QuestionAnswer{}).
		Where("user_id = ?", userID).
		Update("created_at", time.Now().Add(-time.Hour))

	isSpam, err = s.CheckSpam(ctx, userID)
	require.NoError(t, err)
	assert.False(t, isSpam)
}

// utility: получаем db из store через reflect (легально для тестов)
func getInternalDB(t *testing.T, s *storage.Store) *gorm.DB {
	t.Helper()

	f := reflect.ValueOf(s).Elem().FieldByName("db")
	require.True(t, f.IsValid())
	return f.Interface().(*gorm.DB)
}

// -------------------------------
// TEST: GetHistory
// -------------------------------
func TestGetHistory(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	_, userID, _ := s.EnsureUser(ctx, 300)

	s.AddQuestionAnswer(ctx, userID, "q1", "a1", nil)
	s.AddQuestionAnswer(ctx, userID, "q2", "a2", nil)
	s.AddQuestionAnswer(ctx, userID, "q3", "", nil)

	his, err := s.GetHistory(ctx, userID, 30, 5)
	require.NoError(t, err)

	assert.Len(t, his, 3)
	assert.Equal(t, "q1", his[0].Question)
	assert.Equal(t, "q3", his[2].Question)
}

// -------------------------------
// TEST: FilterHistory
// -------------------------------
func TestFilterHistory(t *testing.T) {
	now := time.Now()

	history := []storage.QuestionAnswer{
		{Question: "q1", Answer: "a1", CreatedAt: now.Add(-10 * time.Minute)},
		{Question: "q2", Answer: "", CreatedAt: now.Add(-9 * time.Minute)},
		{Question: "q3", StopPoint: true, CreatedAt: now.Add(-8 * time.Minute)},
		{Question: "q4", Answer: "a4", CreatedAt: now.Add(-7 * time.Minute)},
		{Question: "q5", Answer: "", CreatedAt: now.Add(-6 * time.Minute)},
	}

	ans, unas := storage.FilterHistory(history)

	assert.Len(t, ans, 1)
	assert.Len(t, unas, 1)
	assert.Equal(t, "q4", ans[0].Question)
	assert.Equal(t, "q5", unas[0].Question)
}

// -------------------------------
// TEST: SetStopPoint
// -------------------------------
func TestSetStopPoint(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	_, userID, _ := s.EnsureUser(ctx, 99)

	s.AddQuestionAnswer(ctx, userID, "q1", "a1", nil)
	s.AddQuestionAnswer(ctx, userID, "q2", "a2", nil)

	err := s.SetStopPoint(ctx, userID, true)
	require.NoError(t, err)

	db := getInternalDB(t, s)

	var last storage.QuestionAnswer
	db.Where("user_id = ?", userID).Order("created_at desc").First(&last)

	assert.True(t, last.StopPoint)
}
