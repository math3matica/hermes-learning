# Security policy

This project processes user-provided source material and may call a host-owned
LLM. Do not commit credentials, provider responses, source books, transcripts,
generated state, evidence roots, model files, or private profile data.

Source locators are persisted as job data. The host must authorize paths before
enqueueing or executing jobs; a source path is not permission to read arbitrary
files. Keep `HERMES_REX_LEARNING_HOME` and provider credentials outside the
checkout. Review current files and Git history before publication.

For a suspected security issue, do not include private source material or
credentials in a public issue. Contact the repository owner through a private
channel with a minimal reproduction. This project is not an official Nous
Research component.