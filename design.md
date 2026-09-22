# Lattice Studio UI Design Tokens

This file is the visual contract for the Qt workbench. New screens should use
these tokens instead of introducing local colors or spacing values.

## Color

| Token | Value | Use |
| --- | --- | --- |
| background | `#17181D` | application background |
| navigation | `#1D1F25` | navigation and side panels |
| surface | `#202229` | normal groups and panels |
| surface-raised | `#262831` | elevated controls and hover states |
| surface-active | `#2D303A` | selected controls |
| border | `#353842` | default separators |
| border-strong | `#484C59` | emphasized separators |
| primary | `#7C3AED` | primary action and selection |
| primary-hover | `#8B5CF6` | primary hover |
| primary-pressed | `#6D28D9` | primary pressed state |
| focus | `#A78BFA` | keyboard focus |
| accent | `#0891B2` | secondary action and information |
| accent-hover | `#0E7490` | secondary hover |
| text-primary | `#F4F4F6` | headings and important values |
| text-secondary | `#C2C4CC` | normal body text |
| text-muted | `#8F929E` | helper text and inactive labels |
| success | `#22C55E` | successful operation |
| warning | `#F59E0B` | warning state |
| danger | `#EF4444` | destructive action and error |
| viewport-border | `#C8CBD2` | CAD viewport boundary |

## Typography

Use `Segoe UI`, falling back to `Microsoft YaHei UI` on Windows. Body text is
13 px at normal weight; helper text is 12 px; section labels are 13 px at
600 weight; page titles are 18 px at 600 weight. Keep line height around 1.35
for multiline labels.

## Spacing

Use a 4 px base grid. Standard control padding is 8 px horizontally and 6 px
vertically. Group content uses 10-12 px padding. Keep at least 8 px between
related controls and 16 px between unrelated groups. Avoid dense borders between
every individual field.

## Components

Groups and tabs use a restrained 5-6 px radius, a 1 px `border`, and no
decorative shadow. Primary buttons use `primary`, with a clear hover and pressed
state; destructive buttons use `danger`. Disabled controls use muted text and
the normal surface rather than reducing the entire panel's opacity. The viewport
is an unframed work area with a light CAD background and a thin neutral border.
