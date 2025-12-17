package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/AI2HU/gego/internal/llm"
	"github.com/AI2HU/gego/internal/models"
)

// Provider implements the LLM Provider interface for OpenAI
type Provider struct {
	apiKey            string
	baseURL           string
	client            openai.Client
	systemInstruction string
}

// New creates a new OpenAI provider
func New(apiKey, baseURL, systemInstruction string) *Provider {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	if baseURL != "" && baseURL != "https://api.openai.com/v1" {
		client = openai.NewClient(
			option.WithAPIKey(apiKey),
			option.WithBaseURL(baseURL),
		)
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
	return "openai"
}

// Validate validates the provider configuration
func (p *Provider) Validate(config map[string]string) error {
	if config["api_key"] == "" {
		return fmt.Errorf("api_key is required")
	}
	return nil
}

// Generate sends a prompt to OpenAI and returns the response
func (p *Provider) Generate(ctx context.Context, prompt string, config llm.Config) (*llm.Response, error) {
	model := shared.ChatModelGPT3_5Turbo
	if config.Model != "" {
		model = shared.ChatModel(config.Model)
	}

	temperature := config.Temperature
	maxTokens := config.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1000
	}

	inputText := prompt
	instructions := ""
	if p.systemInstruction != "" {
		instructions = p.systemInstruction
	}

	responseParams := responses.ResponseNewParams{
		Model: responses.ChatModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(inputText),
		},
		Tools: []responses.ToolUnionParam{
			responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch),
		},
		ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: openai.Opt(responses.ToolChoiceOptionsRequired),
		},
		Include: []responses.ResponseIncludable{
			"web_search_call.action.sources",
		},
	}

	if instructions != "" {
		responseParams.Instructions = openai.String(instructions)
	}

	modelStr := strings.ToLower(string(model))
	supportsTemperature := !strings.HasPrefix(modelStr, "gpt-5") && !strings.HasPrefix(modelStr, "o1") && !strings.HasPrefix(modelStr, "o3")

	if temperature > 0 && supportsTemperature {
		responseParams.Temperature = openai.Float(temperature)
	}

	if maxTokens > 0 {
		responseParams.MaxOutputTokens = openai.Int(int64(maxTokens))
	}

	resp, err := p.client.Responses.New(ctx, responseParams)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API error: %w", err)
	}

	generatedText := resp.OutputText()

	var searchURLs []llm.SearchURL
	for _, outputItem := range resp.Output {
		if outputItem.Type == "web_search_call" {
			searchQuery := llm.UnknownSearchQuery
			if outputItem.Action.Query != "" {
				searchQuery = outputItem.Action.Query
			}

			if outputItem.Action.Type == "search" || outputItem.Action.Type == "web_search" {
				if len(outputItem.Action.Sources) > 0 {
					for index, source := range outputItem.Action.Sources {
						if source.URL != "" {
							searchURLs = append(searchURLs, llm.SearchURL{
								SearchQuery:   searchQuery,
								URL:           source.URL,
								Title:         "",
								CitationIndex: index,
							})
						}
					}
				}
			}
		}
	}

	tokensUsed := 0
	if resp.Usage.TotalTokens != 0 {
		tokensUsed = int(resp.Usage.TotalTokens)
	}

	return &llm.Response{
		Text:       generatedText,
		TokensUsed: tokensUsed,
		Model:      string(model),
		Provider:   "openai",
		SearchURLs: searchURLs,
	}, nil
}

// ListModels lists available text-to-text models from OpenAI
func (p *Provider) ListModels(ctx context.Context, apiKey, baseURL string) ([]models.ModelInfo, error) {
	client := p.client
	if apiKey != "" && apiKey != p.apiKey {
		client = openai.NewClient(
			option.WithAPIKey(apiKey),
		)
		if baseURL != "" && baseURL != "https://api.openai.com/v1" {
			client = openai.NewClient(
				option.WithAPIKey(apiKey),
				option.WithBaseURL(baseURL),
			)
		}
	}

	modelList, err := client.Models.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	var textModels []models.ModelInfo
	var popularModels []models.ModelInfo
	popularModelIDs := map[string]bool{
		"gpt-3.5-turbo": true,
		"gpt-4":         true,
		"gpt-4-turbo":   true,
		"gpt-4o":        true,
		"gpt-4o-mini":   true,
	}

	for _, model := range modelList.Data {
		modelID := string(model.ID)

		if strings.HasPrefix(strings.ToLower(modelID), "gpt") {
			if strings.Contains(modelID, ":") {
				continue
			}

			if strings.Contains(strings.ToLower(modelID), "embed") || strings.Contains(strings.ToLower(modelID), "embedding") {
				continue
			}

			if strings.Contains(strings.ToLower(modelID), "vision") || strings.Contains(strings.ToLower(modelID), "image") {
				continue
			}

			if strings.Contains(strings.ToLower(modelID), "whisper") || strings.Contains(strings.ToLower(modelID), "audio") {
				continue
			}

			info := models.ModelInfo{
				ID:          modelID,
				Name:        modelID,
				Description: fmt.Sprintf("OpenAI %s", modelID),
				UsedInChat:  popularModelIDs[modelID],
			}

			if popularModelIDs[modelID] {
				popularModels = append(popularModels, info)
			} else {
				textModels = append(textModels, info)
			}
		}
	}

	return append(popularModels, textModels...), nil
}
