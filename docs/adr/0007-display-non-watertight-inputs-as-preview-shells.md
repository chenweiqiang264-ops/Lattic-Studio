# Display non-watertight inputs as preview shells

Non-watertight STL inputs may be converted to a display-only implicit shell so users can inspect them without waiting for repair. The shell is explicitly non-authoritative: it cannot participate in TPMS boolean intersection or STL export until the source passes watertightness checks.
