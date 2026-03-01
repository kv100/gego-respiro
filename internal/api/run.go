package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/AI2HU/gego/internal/db"
	"github.com/AI2HU/gego/internal/llm"
	"github.com/AI2HU/gego/internal/llm/anthropic"
	"github.com/AI2HU/gego/internal/llm/google"
	"github.com/AI2HU/gego/internal/llm/ollama"
	"github.com/AI2HU/gego/internal/llm/openai"
	"github.com/AI2HU/gego/internal/llm/perplexity"
	"github.com/AI2HU/gego/internal/services"
)

func refreshLLMRegistry(ctx context.Context, database db.Database) *llm.Registry {
	registry := llm.NewRegistry()

	llms, err := database.ListLLMs(ctx, nil)
	if err != nil {
		return registry
	}

	for _, llmCfg := range llms {
		if !llmCfg.Enabled {
			continue
		}
		var provider llm.Provider
		switch llmCfg.Provider {
		case "openai":
			provider = openai.New(llmCfg.APIKey, llmCfg.BaseURL, "")
		case "anthropic":
			provider = anthropic.New(llmCfg.APIKey, llmCfg.BaseURL)
		case "ollama":
			provider = ollama.New(llmCfg.BaseURL)
		case "google":
			provider = google.New(llmCfg.APIKey, llmCfg.BaseURL, "")
		case "perplexity":
			provider = perplexity.New(llmCfg.APIKey, llmCfg.BaseURL)
		}
		if provider != nil {
			registry.Register(provider)
		}
	}

	return registry
}

func (s *Server) runExecution(c *gin.Context) {
	ctx := c.Request.Context()

	registry := refreshLLMRegistry(ctx, s.db)
	providers := registry.List()
	if len(providers) == 0 {
		s.errorResponse(c, http.StatusInternalServerError, "No LLM providers available. Create one via POST /api/v1/llms first.")
		return
	}

	executionService := services.NewExecutionService(s.db, registry)
	config := &services.ExecutionConfig{
		Temperature: 0.7,
		MaxRetries:  3,
		RetryDelay:  30 * time.Second,
	}

	result, err := executionService.ExecuteAllEnabledPrompts(ctx, config)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, fmt.Sprintf("Execution failed: %v", err))
		return
	}

	s.successResponse(c, gin.H{
		"total":      result.TotalExecutions,
		"successful": result.SuccessfulExecutions,
		"failed":     result.FailedExecutions,
		"duration":   result.EndTime.Sub(result.StartTime).String(),
	})
}
