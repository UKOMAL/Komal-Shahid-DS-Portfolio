# Week 1 Discussion — Post 3 (Reply): Ethics in Data Science Work

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 1 | **Type:** Reply (~200 words)

---

[Reply to classmate discussing ethics and responsible use of data]

The ethics angle you raised is one I think about constantly, especially working with sensitive data. My current project uses mental health survey responses, which is exactly the kind of dataset where the analysis needs to stay at the aggregate level — the moment you start making individual-level predictions from self-reported mental health data, you have crossed into territory that could harm real people even with the best intentions.

What I have found useful is treating ethical review not as a paragraph you write in the proposal and never revisit, but as a recurring checkpoint. Before building a new feature, I ask whether it could function as a proxy for something protected. Before finalizing a visualization, I ask whether a non-expert reader could misread it as diagnostic or predictive of an individual. Those habits slow you down slightly but they make the final work more defensible.

Your point about bias in data collection is also important — if the people most affected by a problem are least likely to appear in your dataset, no amount of modeling corrects that gap. It has to be addressed in how you frame the limitations. That kind of transparency is what makes a white paper trustworthy.

**Reference:**  
American Psychological Association. (2023). *2023 Work in America Survey: Workplaces as engines of psychological health and well-being*. https://www.apa.org/pubs/reports/work-in-america/2023-workplace-health-well-being
