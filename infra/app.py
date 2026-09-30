#!/usr/bin/env python3
"""CDK app entry (see cdk.json). Synth only: `npx -y aws-cdk@2 synth` - never deploy without a cost review."""

import aws_cdk as cdk

from fintech_stack import FinTechAIStack

app = cdk.App()
# Environment-agnostic on purpose: no account/region lookups, so synth needs no AWS credentials.
FinTechAIStack(app, "FinTechAIStack", description="FinTech AI production target architecture (Bedrock).")
app.synth()
