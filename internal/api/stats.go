package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/AI2HU/gego/internal/models"
	"github.com/AI2HU/gego/internal/services"
)

// getStats handles GET /api/v1/stats
func (s *Server) getStats(c *gin.Context) {
	totalResponses, err := s.statsService.GetTotalResponses(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get total responses: "+err.Error())
		return
	}

	totalPrompts, err := s.statsService.GetTotalPrompts(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get total prompts: "+err.Error())
		return
	}

	totalLLMs, err := s.statsService.GetTotalLLMs(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get total LLMs: "+err.Error())
		return
	}

	totalSchedules, err := s.statsService.GetTotalSchedules(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get total schedules: "+err.Error())
		return
	}

	limitStr := c.DefaultQuery("keyword_limit", "10")
	keywordLimit, _ := strconv.Atoi(limitStr)
	if keywordLimit <= 0 || keywordLimit > 100 {
		keywordLimit = 10
	}

	topKeywords, err := s.statsService.GetTopKeywords(c.Request.Context(), keywordLimit, nil, nil)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get top keywords: "+err.Error())
		return
	}

	promptStats, err := s.statsService.GetAllPromptStats(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get prompt stats: "+err.Error())
		return
	}

	llmStats, err := s.statsService.GetAllLLMStats(c.Request.Context())
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get LLM stats: "+err.Error())
		return
	}

	endTime := time.Now()
	startTime := endTime.AddDate(0, 0, -30)
	responseTrends, err := s.statsService.GetResponseTrends(c.Request.Context(), startTime, endTime)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get response trends: "+err.Error())
		return
	}

	response := models.StatsResponse{
		TotalResponses: totalResponses,
		TotalPrompts:   totalPrompts,
		TotalLLMs:      totalLLMs,
		TotalSchedules: totalSchedules,
		TopKeywords:    topKeywords,
		PromptStats:    promptStats,
		LLMStats:       llmStats,
		ResponseTrends: responseTrends,
		LastUpdated:    time.Now(),
	}

	s.successResponse(c, response)
}

type URLStatsResponse struct {
	TopURLs    []*services.URLMentionStats    `json:"top_urls"`
	TopDomains []*services.DomainMentionStats `json:"top_domains"`
}

func (s *Server) getURLStats(c *gin.Context) {
	limitStr := c.DefaultQuery("limit", "20")
	limit, _ := strconv.Atoi(limitStr)
	if limit <= 0 || limit > 100 {
		limit = 20
	}

	topURLs, err := s.statsService.GetTopURLsByCitations(c.Request.Context(), limit)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get top URLs: "+err.Error())
		return
	}

	topDomains, err := s.statsService.GetTopDomainsByCitations(c.Request.Context(), limit)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get top domains: "+err.Error())
		return
	}

	s.successResponse(c, URLStatsResponse{
		TopURLs:    topURLs,
		TopDomains: topDomains,
	})
}

func (s *Server) getQueryURLStats(c *gin.Context) {
	limitStr := c.DefaultQuery("limit", "20")
	limit, _ := strconv.Atoi(limitStr)
	if limit <= 0 || limit > 100 {
		limit = 20
	}

	stats, err := s.statsService.GetQueryURLRelationships(c.Request.Context(), limit)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get query URL relationships: "+err.Error())
		return
	}

	s.successResponse(c, stats)
}

func (s *Server) getKeywordDomainMatrix(c *gin.Context) {
	keywordLimitStr := c.DefaultQuery("keyword_limit", "20")
	domainLimitStr := c.DefaultQuery("domain_limit", "10")

	keywordLimit, _ := strconv.Atoi(keywordLimitStr)
	if keywordLimit <= 0 || keywordLimit > 200 {
		keywordLimit = 20
	}

	domainLimit, _ := strconv.Atoi(domainLimitStr)
	if domainLimit <= 0 || domainLimit > 100 {
		domainLimit = 10
	}

	stats, err := s.statsService.GetKeywordDomainMatrix(c.Request.Context(), keywordLimit, domainLimit)
	if err != nil {
		s.errorResponse(c, http.StatusInternalServerError, "Failed to get keyword-domain matrix: "+err.Error())
		return
	}

	s.successResponse(c, stats)
}

// healthCheck handles GET /api/v1/health
func (s *Server) healthCheck(c *gin.Context) {
	if err := s.db.Ping(c.Request.Context()); err != nil {
		c.JSON(http.StatusServiceUnavailable, models.APIResponse{
			Success: false,
			Error:   "Database connection failed",
		})
		return
	}

	c.JSON(http.StatusOK, models.APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now(),
			"version":   "1.0.0",
		},
	})
}
