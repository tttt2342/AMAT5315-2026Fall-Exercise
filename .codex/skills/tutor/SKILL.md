---
name: tutor
description: Turn a local lesson file or web lesson address into a Chinese-first guided tutoring session, especially AMAT5315 weekly PDF sheets.
---

# Tutor

Use this project skill only in this repository for AMAT5315 computation course exercises and projects.

## Purpose

Turn a lesson from a local file or web address into a guided tutoring session for ZHANG Tianyi. Teach mainly in Chinese while preserving important English technical terms. Prefer explanation over quizzing: build a clear mental model first, then invite questions and check understanding lightly when useful.

The AMAT5315 weekly sheets are PDFs under:

```text
https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/
```

## Lesson Intake

When the user provides a lesson source, identify whether it is:

- a local file path
- a web address
- a week reference that likely maps to a PDF under the weekly sheets URL

If the lesson source is missing or ambiguous, ask for the file, URL, or week number before tutoring.

For non-PDF local text-like files, read the file directly. For non-PDF web pages, use the available browsing tools and cite the page if web access was needed.

## PDF Requirement

When the lesson is a PDF, download or copy it into the repository cache before tutoring, then extract its text with the installed `pypdf` package. Prefer the bundled workspace Python when available because it includes `pypdf`:

```text
/Users/tianyizhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Use [scripts/extract_pdf_text.py](scripts/extract_pdf_text.py) for PDF lessons:

```bash
/Users/tianyizhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .codex/skills/tutor/scripts/extract_pdf_text.py <lesson-pdf-or-url> --output .codex/lesson-cache/<lesson-name>.txt
```

The script caches PDFs in `.codex/lesson-cache/` by default and prints or writes extracted text. If the bundled Python path is unavailable, use another Python environment only after confirming that `import pypdf` works.

## Tutoring Style

After extracting or reading the lesson:

- Start with a short map of the lesson: topic, prerequisites, main definitions, and what the student should be able to do.
- Explain in Chinese, retaining key English terms such as algorithm names, theorem names, API names, and notation.
- Connect abstractions to the course exercises in `week1/`, `week2/`, and so on when relevant.
- Work through important formulas, code ideas, or derivations step by step.
- Keep questions light and purposeful; do not turn the session into a quiz unless the user asks.
- If the lesson is too long, chunk it by section and tutor one section at a time.

## Weekly Notes

When useful or when the user asks, summarize the session as weekly notes. Keep notes concise and course-oriented:

- core concepts
- important formulas or algorithms
- implementation reminders
- common mistakes
- links back to the relevant week folder or lesson source

Only create or update note files when the user asks for a saved artifact. Otherwise provide the notes in the conversation.
