package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
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

	messages := []openai.ChatCompletionMessageParamUnion{}

	if p.systemInstruction != "" {
		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(p.systemInstruction),
				},
			},
		})
	}

	messages = append(messages, openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfString: openai.String(prompt),
			},
		},
	})

	webSearchOptions := openai.ChatCompletionNewParamsWebSearchOptions{
		SearchContextSize: "medium",
	}

	chatCompletion, err := p.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model:            model,
			Messages:         messages,
			WebSearchOptions: webSearchOptions,
			Temperature:      openai.Float(temperature),
			MaxTokens:        openai.Int(int64(maxTokens)),
		},
	)
	if err != nil {
		return nil, fmt.Errorf("OpenAI API error: %w", err)
	}

	var generatedText string
	if len(chatCompletion.Choices) > 0 && chatCompletion.Choices[0].Message.Content != "" {
		generatedText = chatCompletion.Choices[0].Message.Content
	}

	var searchURLs []llm.SearchURL
	if len(chatCompletion.Choices) > 0 {
		message := chatCompletion.Choices[0].Message
		for index, annotation := range message.Annotations {
			if annotation.URLCitation.URL != "" {
				searchURLs = append(searchURLs, llm.SearchURL{
					SearchQuery:   "unknown",
					URL:           annotation.URLCitation.URL,
					Title:         annotation.URLCitation.Title,
					CitationIndex: index,
				})
			}
		}
	}

	tokensUsed := 0
	if chatCompletion.Usage.TotalTokens != 0 {
		tokensUsed = int(chatCompletion.Usage.TotalTokens)
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

			textModels = append(textModels, models.ModelInfo{
				ID:          modelID,
				Name:        modelID,
				Description: fmt.Sprintf("OpenAI %s", modelID),
			})
		}
	}

	return textModels, nil
}
