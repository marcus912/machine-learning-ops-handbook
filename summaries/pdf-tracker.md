# ML/AI PDF Library - Summary Tracker

Track progress on summarizing resources in `/resources/` folder.

---

## Summarization Instructions

### File Locations
- **PDF Source:** `/Users/marcus/dev/workspaces/resume/technical/machine-learning-ops-handbook/resources/`
- **Output Directory:** `/Users/marcus/dev/workspaces/resume/technical/machine-learning-ops-handbook/summaries/`
- **Naming Convention:** `{pdf-name-lowercase-kebab}-summary.md`

### Steps to Summarize a PDF

1. **Extract text using pdftotext** (poppler is installed):
   ```bash
   # Extract first 20 pages to understand structure/TOC
   pdftotext -f 1 -l 20 "/path/to/book.pdf" - | head -500

   # Extract middle sections for core content
   pdftotext -f 100 -l 130 "/path/to/book.pdf" - | head -600

   # Extract later chapters for advanced topics
   pdftotext -f 200 -l 250 "/path/to/book.pdf" - | head -600
   ```

2. **Read multiple sections** to cover:
   - Table of Contents (pages 1-20)
   - Core chapters (middle of book)
   - Advanced topics (later chapters)
   - Conclusion/summary sections

3. **Create summary file** with this structure:
   - Book metadata (author, publisher, pages, focus)
   - Core definition/concept
   - Section-by-section breakdown
   - Key tools & technologies
   - Best practices summary
   - Target audience

4. **Update this tracker** - mark the PDF as complete with `[x]`

### Example Command Sequence
```bash
# Check page count
pdfinfo "/path/to/book.pdf" | grep Pages

# Extract sections
pdftotext -f 1 -l 20 "/path/to/book.pdf" - | head -500    # TOC
pdftotext -f 50 -l 80 "/path/to/book.pdf" - | head -600   # Early chapters
pdftotext -f 150 -l 180 "/path/to/book.pdf" - | head -600 # Mid chapters
pdftotext -f 250 -l 280 "/path/to/book.pdf" - | head -600 # Late chapters
```

---

## Summary Status

| Status | PDF | Pages | Summary File |
|--------|-----|-------|--------------|
| [x] | Engineering_MLOps.pdf | 370 | `engineering-mlops-summary.md` |
| [x] | Machine_Learning_in_Production.pdf | 462 | `machine-learning-in-production-summary.md` |
| [x] | Managing_Machine_Learning_Projects.pdf | 273 | `managing-machine-learning-projects-summary.md` |
| [x] | Cloud_FinOps.pdf | 457 | `cloud-finops-summary.md` |
| [x] | Mastering_TensorFlow_2.x.pdf | 395 | `mastering-tensorflow-2x-summary.md` |
| [x] | A course in ML.pdf | 227 | `a-course-in-ml-summary.md` |
| [x] | A Primer to the 42 Most commonly used Machine Learning Algor.pdf | 192 | `42-ml-algorithms-primer-summary.md` |
| [x] | A Course in Natural Language Processing.pdf | 543 | `a-course-in-nlp-summary.md` |
| [x] | A Gentle Introduction to Quantum Machine Learning 2025.pdf | 215 | `quantum-ml-gentle-intro-summary.md` |
| [x] | _Human_in_the_Loop_Machine_Learning_Active_learning_and_annotation.pdf | 426 | `human-in-the-loop-machine-learning-summary.md` |
| [x] | _media_books_machine_learning_with_python_theory_and_applications.pdf | 693 | `ml-with-python-theory-applications-summary.md` |
| [x] | Blockchain_Tethered_AI.pdf | 307 | `blockchain-tethered-ai-summary.md` |
| [x] | Finite_Difference_Computing_with_Exponential_Decay_Models.pdf | 210 | `finite-difference-exponential-decay-summary.md` |
| [x] | 2021912185653678BasicsofLinearAlgebraforMachineLearningbyJasonBrownlee.pdf | 212 | `linear-algebra-for-ml-brownlee-summary.md` |
| [x] | machine-learning-design-patterns-solutions-to-common-challenges-in-data-preparation-model-building-and-mlops-1098115783-9781098115784_compress.pdf | 408 | `ml-design-patterns-summary.md` |

---

## Priority Queue

### High Priority (Core MLOps/ML Engineering)
1. [x] Machine_Learning_in_Production.pdf
2. [x] Managing_Machine_Learning_Projects.pdf
3. [x] Cloud_FinOps.pdf
4. [x] machine-learning-design-patterns-solutions-to-common-challenges-in-data-preparation-model-building-and-mlops-1098115783-9781098115784_compress.pdf

### Medium Priority (ML Foundations)
4. [x] Mastering_TensorFlow_2.x.pdf
5. [x] A course in ML.pdf
6. [x] A Primer to the 42 Most commonly used Machine Learning Algor.pdf
7. [x] _Human_in_the_Loop_Machine_Learning_Active_learning_and_annotation.pdf

### Lower Priority (Specialized Topics)
8. [x] A Course in Natural Language Processing.pdf
9. [x] A Gentle Introduction to Quantum Machine Learning 2025.pdf
10. [x] Blockchain_Tethered_AI.pdf
11. [x] _media_books_machine_learning_with_python_theory_and_applications.pdf
12. [x] Finite_Difference_Computing_with_Exponential_Decay_Models.pdf
13. [x] 2021912185653678BasicsofLinearAlgebraforMachineLearningbyJasonBrownlee.pdf

---

## Summary Template

Use this markdown structure for each summary file:

```markdown
# {Book Title} - Comprehensive Summary

**Author:** {Author Name}
**Publisher:** {Publisher}, {Year}
**Pages:** {Page Count}
**Focus:** {One-line description}

---

## Core Definition
> Key quote or concept from the book

---

## Section 1: {Section Title}
### Chapter X: {Chapter Title}
- Key points with **bold** for important terms
- Tables for comparisons
- Code blocks for examples

---

## Key Tools & Technologies
| Category | Tools |
|----------|-------|
| ... | ... |

---

## Best Practices Summary
1. **Practice Name**: Description
2. ...

---

## Target Audience
- Role 1
- Role 2
```

---

## Progress

- **Total PDFs:** 15
- **Completed:** 15
- **Remaining:** 0
- **Completion:** 100%
