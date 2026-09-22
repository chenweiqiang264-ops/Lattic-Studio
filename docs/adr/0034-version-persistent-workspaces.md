# Version persistent workspaces

Persistent workspace metadata will carry an explicit schema version and be loaded through version-specific migrations. New releases may change internal domain and application models while continuing to open older saved designs; disposable sampled fields and renderer caches remain outside the compatibility contract.

