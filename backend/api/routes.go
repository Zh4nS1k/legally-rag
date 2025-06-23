// routes.go

package api

import (
	"legally/api/controllers"
	"legally/api/middleware"
	"legally/db"
	"legally/models"

	"github.com/gin-gonic/gin"
)

func SetupRoutes(router *gin.Engine) {
	router.Use(middleware.CORSMiddleware())
	router.Use(middleware.LoggerMiddleware())

	// Статика и корневая страница
	router.Static("/static", "./public")
	router.GET("/", func(c *gin.Context) {
		c.File("./public/index.html")
	})

	// Health check
	router.GET("/health", func(c *gin.Context) {
		if err := db.Ping(); err != nil {
			c.JSON(503, gin.H{"status": "unhealthy"})
			return
		}
		c.JSON(200, gin.H{"status": "healthy"})
	})

	// Публичные маршруты
	public := router.Group("/api")
	{
		public.POST("/register", controllers.Register)
		public.POST("/login", controllers.Login)
		public.POST("/refresh", controllers.Refresh)
		public.GET("/validate-token", controllers.ValidateToken)
		public.GET("/laws", controllers.GetRelevantLaws)
	}

	private := router.Group("/api")
	private.Use(middleware.AuthRequired(models.RoleUser))
	{
		private.POST("/analyze", controllers.AnalyzeDocument)
		private.GET("/history", controllers.GetHistory)
		private.POST("/logout", controllers.Logout)
		private.GET("/user", controllers.GetUser)
		private.POST("/analysis/cancel", controllers.CancelAnalysis)
		private.POST("/cache/clear", controllers.ClearFileCache)
		private.POST("/similar", controllers.FindSimilarDocuments) // 🔍 Новый эндпоинт
	}

	// Админские маршруты
	admin := router.Group("/api/admin")
	admin.Use(middleware.AuthRequired(models.RoleAdmin))
	{
		// TODO: admin endpoints
	}
}