package google

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/AI2HU/gego/internal/llm"
	"github.com/AI2HU/gego/internal/models"
)

// Provider implements the LLM Provider interface for Google AI
type Provider struct {
	apiKey            string
	baseURL           string
	client            *genai.Client
	systemInstruction string
}

// New creates a new Google provider
func New(apiKey, baseURL, systemInstruction string) *Provider {
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		client = nil
	}

	return &Provider{
		apiKey:            apiKey,
		baseURL:           baseURL,
		client:            client,
		systemInstruction: systemInstruction,
	}
}

// Name returns the provider name
func (p *Provider) Name() string {
	return "google"
}

// Validate validates the provider configuration
func (p *Provider) Validate(config map[string]string) error {
	if config["api_key"] == "" {
		return fmt.Errorf("api_key is required")
	}
	return nil
}

// Generate sends a prompt to Google AI and returns the response
func (p *Provider) Generate(ctx context.Context, prompt string, config llm.Config) (*llm.Response, error) {
	model := "gemini-1.5-flash"
	if config.Model != "" {
		model = config.Model
	}

	maxTokens := int32(config.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 1000
	}

	var systemInstructionContent *genai.Content
	if p.systemInstruction != "" {
		systemInstructionContent = &genai.Content{
			Parts: []*genai.Part{
				{Text: p.systemInstruction},
			},
		}
	}

	safetySettings := []*genai.SafetySetting{
		{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockMediumAndAbove},
		{Category: genai.HarmCategoryHateSpeech, Threshold: genai.HarmBlockThresholdBlockMediumAndAbove},
		{Category: genai.HarmCategorySexuallyExplicit, Threshold: genai.HarmBlockThresholdBlockMediumAndAbove},
		{Category: genai.HarmCategoryDangerousContent, Threshold: genai.HarmBlockThresholdBlockMediumAndAbove},
		{Category: genai.HarmCategoryCivicIntegrity, Threshold: genai.HarmBlockThresholdBlockMediumAndAbove},
	}

	client := p.client
	if client == nil {
		var err error
		client, err = genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:  p.apiKey,
			Backend: genai.BackendGeminiAPI,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create Google client: %w", err)
		}
	}

	content := []*genai.Content{
		{
			Parts: []*genai.Part{
				{Text: prompt},
			},
		},
	}

	generationConfig := &genai.GenerateContentConfig{
		SystemInstruction: systemInstructionContent,
		Temperature:       float32Ptr(float32(config.Temperature)),
		TopP:              float32Ptr(float32(config.TopP)),
		TopK:              float32Ptr(float32(config.TopK)),
		MaxOutputTokens:   maxTokens,
		SafetySettings:    safetySettings,
		ResponseModalities: []string{
			"TEXT",
		},
		Tools: []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
		},
	}

	result, err := client.Models.GenerateContent(ctx, model, content, generationConfig)
	if err != nil {
		return nil, fmt.Errorf("google AI API error: %v", err)
	}

	var generatedText string
	if len(result.Candidates) > 0 && len(result.Candidates[0].Content.Parts) > 0 {
		if text := result.Candidates[0].Content.Parts[0].Text; text != "" {
			generatedText = text
		}
	}

	var searchURLs []llm.SearchURL
	if len(result.Candidates) > 0 {
		candidate := result.Candidates[0]

		var searchQuery string
		if candidate.GroundingMetadata != nil && len(candidate.GroundingMetadata.WebSearchQueries) > 0 {
			searchQuery = candidate.GroundingMetadata.WebSearchQueries[0]
		}

		if candidate.GroundingMetadata != nil {
			for index, chunk := range candidate.GroundingMetadata.GroundingChunks {
				if chunk.Web != nil && chunk.Web.URI != "" {
					resolvedURL := resolveRedirectURL(ctx, chunk.Web.URI)
					searchURLs = append(searchURLs, llm.SearchURL{
						SearchQuery:   searchQuery,
						URL:           resolvedURL,
						Title:         chunk.Web.Title,
						CitationIndex: index,
					})
				}
			}
		}

		if candidate.URLContextMetadata != nil {
			groundingChunkCount := 0
			if candidate.GroundingMetadata != nil {
				groundingChunkCount = len(candidate.GroundingMetadata.GroundingChunks)
			}
			for index, urlMeta := range candidate.URLContextMetadata.URLMetadata {
				if urlMeta.RetrievedURL != "" {
					resolvedURL := resolveRedirectURL(ctx, urlMeta.RetrievedURL)
					searchURLs = append(searchURLs, llm.SearchURL{
						SearchQuery:   searchQuery,
						URL:           resolvedURL,
						Title:         "",
						CitationIndex: groundingChunkCount + index,
					})
				}
			}
		}
	}

	tokensUsed := 0
	if result.UsageMetadata != nil {
		tokensUsed = int(result.UsageMetadata.TotalTokenCount)
	}

	return &llm.Response{
		Text:       generatedText,
		TokensUsed: tokensUsed,
		Model:      model,
		Provider:   "google",
		SearchURLs: searchURLs,
	}, nil
}

// ListModels lists available Google AI models
func (p *Provider) ListModels(ctx context.Context, apiKey, baseURL string) ([]models.ModelInfo, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Google client: %w", err)
	}

	modelPage, err := client.Models.List(ctx, &genai.ListModelsConfig{})
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	var modelList []models.ModelInfo
	for _, model := range modelPage.Items {
		modelName := model.Name

		if strings.Contains(strings.ToLower(modelName), "embed") || strings.Contains(strings.ToLower(modelName), "embedding") {
			continue
		}

		if strings.Contains(strings.ToLower(modelName), "vision") || strings.Contains(strings.ToLower(modelName), "image") {
			continue
		}

		if strings.Contains(strings.ToLower(modelName), "gemini") {
			name := modelName
			if len(name) > 7 && name[:7] == "models/" {
				name = name[7:]
			}

			modelList = append(modelList, models.ModelInfo{
				ID:          model.Name,
				Name:        name,
				Description: model.Description,
			})
		}
	}

	return modelList, nil
}

func float32Ptr(f float32) *float32 {
	return &f
}

// resolveRedirectURL follows HTTP redirects to get the actual destination URL
func resolveRedirectURL(ctx context.Context, url string) string {
	if !strings.HasPrefix(url, "https://vertexaisearch.cloud.google.com/grounding-api-redirect/") {
		return url
	}

	client := &http.Client{
		Timeout: 5 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil
		},
	}

	req, err := http.NewRequestWithContext(ctx, "HEAD", url, nil)
	if err != nil {
		return url
	}

	resp, err := client.Do(req)
	if err != nil {
		return url
	}
	defer resp.Body.Close()

	if resp.Request != nil && resp.Request.URL != nil {
		return resp.Request.URL.String()
	}

	return url
}
