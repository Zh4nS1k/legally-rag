// analysis_service
package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"legally/repositories"
	"legally/utils"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/net/context"
)

const (
	model       = "deepseek/deepseek-r1-0528:free"
	apiEndpoint = "https://openrouter.ai/api/v1/chat/completions"
)

type HttpError struct {
	Status  int
	Message string
}

func AnalyzeDocument(c *gin.Context) (interface{}, *HttpError) {
	utils.LogAction("Получен запрос на анализ документа")

	text, filename, err := utils.ProcessUploadedFile(c)
	if err != nil {
		utils.LogError(err.Error())
		return nil, &HttpError{Status: http.StatusBadRequest, Message: err.Error()}
	}

	utils.LogInfo(fmt.Sprintf("Извлечено %d символов из документа", len(text)))

	analysis, docType, err := AnalyzeText(text)
	if err != nil {
		utils.LogError(err.Error())
		return nil, &HttpError{Status: http.StatusInternalServerError, Message: err.Error()}
	}

	userID, _ := c.Get("userId")
	err = repositories.SaveAnalysis(userID.(string), filename, docType, analysis, text)
	if err != nil {
		utils.LogWarning(fmt.Sprintf("Ошибка сохранения в MongoDB: %v", err))
	}

	// 🔵 Отправка эмбеддинга в Pinecone
	err = SendToPinecone(
		userID.(string),
		text,
		map[string]string{
			"filename": filename,
			"type":     docType,
		},
	)
	if err != nil {
		utils.LogWarning(fmt.Sprintf("Ошибка отправки в Pinecone: %v", err))
	}

	utils.LogSuccess("Полный анализ готов, отправляем ответ клиенту")
	utils.LogInfo(fmt.Sprintf("Тип документа: %s, длина анализа: %d символов", docType, len(analysis)))

	return gin.H{
		"analysis":      analysis,
		"timestamp":     time.Now().Format(time.RFC3339),
		"document_type": docType,
		"filename":      filename,
	}, nil
}

func AnalyzeText(text string) (string, string, error) {
	parts := utils.SplitText(text, 12000)
	utils.LogInfo(fmt.Sprintf("Документ разбит на %d частей для анализа", len(parts)))

	var analysisResults []string
	for i, part := range parts {
		partNum := i + 1
		utils.LogAction(fmt.Sprintf("Анализ части %d/%d...", partNum, len(parts)))

		result, err := analyzeDocumentPart(part)
		if err != nil {
			utils.LogError(fmt.Sprintf("При анализе части %d: %v", partNum, err))
			return "", "", err
		}

		utils.LogSuccess(fmt.Sprintf("Анализ части %d завершён, результат длиной %d символов", partNum, len(result)))
		analysisResults = append(analysisResults, result)
	}

	fullAnalysis := strings.Join(analysisResults, "\n\n---\n\n")
	docType := detectDocumentType(text)

	return fullAnalysis, docType, nil
}

func analyzeDocumentPart(text string) (string, error) {
	prompt := fmt.Sprintf(`Проанализируй следующий юридический документ на соответствие законодательству Казахстана. 

В ответе придерживайся следующей структуры:

### Правовые риски

1. [Название риска]
   - Описание: [подробное описание]
   - Нормативный акт: [закон/статья]
   - Уровень риска: [высокий/средний/низкий]
   - Рекомендация: [предложение по исправлению]

2. [Название риска]
   ...

### Неясные формулировки

1. [Формулировка]
   - Проблема: [в чем неясность]
   - Рекомендация: [как переформулировать]
   - Уровень важности: [высокий/средний/низкий]

### Возможные нарушения

1. [Описание нарушения]
   - Нормативный акт: [закон/статья]
   - Последствия: [возможные санкции]
   - Рекомендация: [как избежать]

### Рекомендации

[Список конкретных рекомендаций по исправлению документа]

### Заключение

[Общая сводка по документу с выводами]

Документ:
%s`, text)

	utils.LogInfo(fmt.Sprintf("Отправка запроса к AI с текстом длиной %d символов", len(text)))

	result, err := queryOpenRouter(prompt)
	if err != nil {
		return "", err
	}

	utils.LogSuccess(fmt.Sprintf("Успешно получен ответ от AI длиной %d символов", len(result)))
	return result, nil
}

func queryOpenRouter(prompt string) (string, error) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENROUTER_API_KEY не установлен")
	}

	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": "Ты — юридический эксперт по законодательству Казахстана. Анализируй документы и давай развернутые ответы с конкретными ссылками на законы."},
			{"role": "user", "content": prompt},
		},
		"temperature": 0.3,
		"max_tokens":  4000,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("ошибка маршалинга payload: %w", err)
	}

	utils.LogRequest("out", apiEndpoint, len(body))

	req, err := http.NewRequest("POST", apiEndpoint, bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("ошибка создания запроса: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://legally.kz")
	req.Header.Set("X-Title", "Legally AI Risk Analyzer")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ошибка запроса к OpenRouter: %w", err)
	}
	defer resp.Body.Close()

	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ошибка чтения ответа OpenRouter: %w", err)
	}

	utils.LogRequest("in", fmt.Sprintf("OpenRouter (статус: %d)", resp.StatusCode), len(resBody))

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ошибка от OpenRouter: статус %d", resp.StatusCode)
	}

	var res struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(resBody, &res); err != nil {
		return "", fmt.Errorf("не удалось распарсить ответ AI: %w", err)
	}

	if len(res.Choices) == 0 || res.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("пустой ответ от OpenRouter")
	}

	return res.Choices[0].Message.Content, nil
}

func GetRelevantLaws() []map[string]string {
	return []map[string]string{
		{"name": "Гражданский кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K950001000_"},
		{"name": "Налоговый кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K2100000409"},
		{"name": "Трудовой кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K1500000011"},
		{"name": "Кодекс об административных правонарушениях РК", "url": "https://adilet.zan.kz/rus/docs/K1400000233"},
	}
}

func GetUserHistory(userID string) ([]map[string]interface{}, error) {
	return repositories.GetUserHistory(userID)
}

func detectDocumentType(text string) string {
	text = strings.ToLower(text)
	switch {
	case strings.Contains(text, "договор"):
		return "Договор"
	case strings.Contains(text, "приказ"):
		return "Приказ"
	case strings.Contains(text, "постановление"):
		return "Постановление"
	case strings.Contains(text, "закон"):
		return "Закон"
	case strings.Contains(text, "решение"):
		return "Решение"
	default:
		return "Неизвестно"
	}
}

var (
	userCache      = make(map[string]string)
	activeAnalysis = make(map[string]context.CancelFunc)
	cacheMutex     sync.Mutex
)

func CancelUserAnalysis(userID string) error {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	if cancel, exists := activeAnalysis[userID]; exists {
		cancel()
		delete(activeAnalysis, userID)
		delete(userCache, userID)
		return nil
	}
	return fmt.Errorf("анализ не найден или уже завершен")
}

func ClearUserCache(userID string) error {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	delete(userCache, userID)
	return nil
}

func CacheUserFile(userID, content string) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	userCache[userID] = content
}

func GetCachedFile(userID string) (string, bool) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	content, exists := userCache[userID]
	return content, exists
}

func StartAnalysis(ctx context.Context, userID string, fn func()) context.Context {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	ctx, cancel := context.WithCancel(ctx)
	activeAnalysis[userID] = cancel

	go func() {
		fn()
		cacheMutex.Lock()
		delete(activeAnalysis, userID)
		cacheMutex.Unlock()
	}()

	return ctx
}