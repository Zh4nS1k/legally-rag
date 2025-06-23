// pinecone
package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type EmbedPayload struct {
	UserID   string            `json:"user_id"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata"`
}

func SendToPinecone(userID, text string, metadata map[string]string) error {
	payload := EmbedPayload{
		UserID:   userID,
		Text:     text,
		Metadata: metadata,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := http.Post("http://localhost:8081/embed", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("ошибка отправки в pinecone-сервис: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("pinecone-сервис вернул статус %d", resp.StatusCode)
	}

	return nil
}