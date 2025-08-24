//similarity_controller
package controllers

import (
	"github.com/gin-gonic/gin"
	"legally/services"
	"net/http"
)

func FindSimilarDocuments(c *gin.Context) {
	userID, _ := c.Get("userId")

	var req struct {
		Text string `json:"text"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Неверный формат запроса"})
		return
	}

	results, err := services.SearchSimilar(userID.(string), req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Ошибка поиска: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"matches": results})
}