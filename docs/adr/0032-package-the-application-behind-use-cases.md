# Package the application behind use cases

The production application will live in the `lattice_studio` package and use the dependency direction `presentation -> application -> domain`, with numerical capabilities supplied through engine interfaces and platform-specific implementations supplied by infrastructure adapters. The existing test-hosted workbench will remain only while callers migrate, then be removed; this preserves behaviour during the migration without retaining production code under `tests` or allowing Qt state to remain the system's business model.

