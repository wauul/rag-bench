# Release policy

Release tags are immutable `vX.Y.Z`, match pyproject version and point to a commit on protected main. Do not retag, overwrite an image tag, or publish a PR artifact. Release reruns engineering/PostgreSQL, real CPU/ONNX models and actual container/security checks before reviewer-controlled publication. GHCR version and full-commit tags resolve to recorded digests; deploy by digest where hosting supports it. Preserve manifests/SBOMs/attestations and known-good source environments for rollback.

A published artifact is not a deployed or quality-approved release. Live sample evaluation is separate, explicitly authorized and bounded. Production promotion needs measured independent evidence and human review. Source-hosted Render and Streamlit rebuilds cannot be described as immutable image promotion. PostgreSQL/checkpoint compatibility, approved dashboard branch promotion and authenticated hosted smoke are release requirements.

Version guidance: patch for compatible fixes, minor for additive workflow/API behavior, major for incompatible contracts. Storage/checkpoint migrations require a reviewed recovery path before deployment. No legal license decision is included.
