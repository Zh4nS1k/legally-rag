//search
package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

func SearchSimilar(userID, text string) ([]map[string]interface{}, error) {
	payload := map[string]interface{}{
		"query": text,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("ошибка кодирования запроса: %w", err)
	}

	resp, err := http.Post("http://localhost:8081/search", "application/json", bytes.NewBuffer(body)) // ✅ Укажи актуальный адрес сервиса
	if err != nil {
		return nil, fmt.Errorf("ошибка вызова AI-сервиса: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("AI-сервис вернул %d: %s", resp.StatusCode, msg)
	}

	var res struct {
		Matches []map[string]interface{} `json:"matches"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("не удалось декодировать ответ AI: %w", err)
	}

	return res.Matches, nil
}