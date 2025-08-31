# Notion Product Documentation & Changelog

## Version 3.12 — October 2024

### Bug Fixes
- **FIXED**: iOS crash on databases with >300 entries (affects iPhone iOS 17+). Deploy rolled out Oct 22, 2024.
- **FIXED**: Web clipper Chrome extension compatibility restored after Manifest V3 migration. Fix deployed Oct 20, 2024.
- **IMPROVED**: Page load time reduced by 40% on desktop via lazy-loading of off-screen blocks.

### Known Issues (In Progress)
- Android performance degradation on Android 14 — fix targeted for v3.13 (Nov 2024).
- Slack integration intermittent sync failures — investigating root cause.
- API rate limit increase planned for Q1 2025 (10 req/sec for paid plans).

---

## Performance & Reliability

### Page Load Times
Our engineering team monitors p95 load times continuously. Current targets:
- Desktop web: < 2 seconds for pages < 500 blocks
- Mobile: < 3 seconds cold start

If you experience load times exceeding these, please report via Settings → Help → Report a Problem with your workspace URL.

### Offline Mode
Notion currently supports limited offline mode for recently viewed pages on mobile. Full offline support (including databases) is on our 2025 roadmap. We recommend the desktop app for offline-heavy workflows.

---

## Pricing & Plans

### Current Plans (as of Oct 2024)
- **Free**: Unlimited pages, up to 10 collaborators, limited block history
- **Plus**: $10/user/month (annual) — unlimited history, guests, file uploads
- **Business**: $15/user/month (annual) — SAML SSO, advanced permissions, audit log
- **Enterprise**: Custom pricing — dedicated manager, SLA guarantees

### Notion AI Add-on
$8/user/month (annual) for Plus/Business plans. Included free in Enterprise plans.
AI features include: writing assistant, Q&A across workspace, auto-summarization.

---

## API Documentation

### Rate Limits
- All plans: 3 requests/second per integration
- Q1 2025: Increasing to 10 req/sec for Business and Enterprise plans
- Bulk operations endpoint (beta) available for Enterprise — contact support

### Authentication
All API requests require a Bearer token from your integration settings.
See: https://developers.notion.com/docs/authorization

---

## Mobile Apps

### iOS
Minimum requirement: iOS 16.0+
Latest version: 3.12.1 (released Oct 22, 2024)
Known fix in 3.12.1: database crash on large collections

### Android
Minimum requirement: Android 10+
Latest version: 3.12.0 (performance fix in v3.13, coming Nov 2024)

---

## Integrations

### Slack
The Notion-Slack integration supports: page mentions, database change notifications, slash commands.
Current status: Intermittent notification delays being investigated (Oct 2024).
Workaround: Re-authorize the Slack integration from Settings → Connections.

### Jira
Two-way sync available for Business/Enterprise plans.
Setup guide: https://notion.so/help/jira-integration

---

## Frequently Asked Questions

**Q: Why is Notion slow?**
A: Large workspaces with many blocks can affect load times. Try: (1) breaking large pages into sub-pages, (2) archiving unused databases, (3) using the desktop app for heavy workflows.

**Q: Can I export my data?**
A: Yes — Settings → Settings → Export all workspace content. Supports HTML, Markdown+CSV, and PDF formats.

**Q: Is there a widget for mobile home screen?**
A: Home screen widgets are on our roadmap for H1 2025. Currently available: lock screen widget (iOS 16+) for quick capture.
