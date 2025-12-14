# Security notes for local configuration

- Use `pipeline_config.example.yaml` as the starting point. Copy it to
  `pipeline_config.local.yaml` and adjust paths or diarization credentials on
  your machine. The local file is ignored by git.
- Provide Hugging Face credentials via the `HF_TOKEN` environment variable when
  possible. The `hf_token` fields in both `pipeline_config.yaml` and the example
  file are placeholders and should remain non-secret values (`HF_TOKEN` or
  empty strings).
- Keep real tokens, API keys, and machine-specific overrides out of version
  control. Restrict sharing of filled-in local configs to trusted teammates and
  prefer environment variables for secrets during automation.
