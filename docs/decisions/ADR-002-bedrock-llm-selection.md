# ADR-002: Amazon Bedrock for LLM Services

## Status: Accepted

## Context
We needed to select an LLM platform for financial analysis that meets regulatory compliance requirements, provides enterprise-grade security, and offers strong reasoning capabilities for complex financial scenarios.

## Decision
We chose Amazon Bedrock with Claude 3 Sonnet for LLM-powered financial analysis and reasoning tasks.

## Alternatives Considered
- **OpenAI GPT-4**: Direct API access with strong reasoning capabilities
- **Anthropic Claude Direct**: Direct access to Claude models via API
- **Azure OpenAI**: Microsoft's hosted OpenAI services
- **Self-hosted Open Source**: Llama 2, Code Llama, or other open models

## Consequences

### Positive
- **Regulatory Compliance**: Data stays within AWS environment meeting SEC/FINRA requirements
- **Security**: Built-in encryption, IAM integration, and audit trails
- **Enterprise SLA**: 99.9% uptime guarantees with enterprise support
- **Cost Predictability**: Integrated AWS billing with detailed usage tracking
- **Guardrails**: Built-in content filtering for financial advice compliance
- **Regional Control**: Data processing within specified AWS regions

### Negative
- **Model Limitations**: Limited to Bedrock-supported models and versions
- **Vendor Lock-in**: Deeper integration with AWS ecosystem increases switching costs
- **Feature Lag**: New model capabilities may arrive later than direct APIs

### Neutral
- **Performance**: Comparable latency to direct APIs (2-3 seconds for complex analysis)
- **Cost**: Similar pricing to direct API calls (~$20/1M tokens)
- **Integration**: Native AWS SDK support simplifies development

### Compliance Benefits
- **Model Risk Management**: Meets SR 11-7 requirements for model governance
- **Data Lineage**: Complete audit trail for all LLM interactions
- **Access Control**: Granular IAM policies for different user roles
- **Data Residency**: Ensures sensitive financial data doesn't cross jurisdictions

### Financial Use Cases
- **Market Analysis**: Sector rotation analysis and thematic investment research
- **Risk Assessment**: Scenario analysis and stress testing narratives
- **Portfolio Commentary**: Performance attribution and investment rationale
- **Compliance Reporting**: Automated regulatory report generation