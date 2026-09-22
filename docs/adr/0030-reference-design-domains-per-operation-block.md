# Reference design domains per operation block

Accepted target for the block-workflow migration; not yet implemented. A Design
document may own multiple Design domains, and each lattice block explicitly
references its filling domain, allowing independent modeling branches within
one document while retaining document-local references and state isolation.
This replaces ADR-0029's one-domain-per-document restriction; the rest of that
decision remains applicable except where later workflow decisions supersede it.

The trade-off is a broader migration from document-global domain state to
explicit block inputs, in exchange for reusable, composable operations without
implicit changes to unrelated modeling branches. How transitions combine
operands with different domains remains a separate decision.
