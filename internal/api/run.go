package api

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/AI2HU/gego/internal/services"
)

func (s *Server) runExecution(c *gin.Context) {
	if s.llmRegistry == nil {
		s.errorResponse(c, http.StatusInternalServerError, "LLM registry not initialized")
		return
	}

	ctx := c.Request.Context()

	executionService := services.NewExecutionService(s.db, s.llmRegistry)
	config := &services.ExecutionConfig{
		Temperature: 0.7,
		MaxRetries:  3,
		RetryDelay:  30 * time.Second,
	}

	result, err := executionService.ExecuteAllEnabledPrompts(ctx, config)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Execution failed: "+err.Error())
		return
	}

	s.successResponse(c, gin.H{
		"total":      result.TotalExecutions,
		"successful": result.SuccessfulExecutions,
		"failed":     result.FailedExecutions,
		"duration":   result.EndTime.Sub(result.StartTime).String(),
	})
}
