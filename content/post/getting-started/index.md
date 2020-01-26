---
authors:
- admin
categories:
- Data Science
date: "2020-01-26T00:00:00Z"
draft: false
featured: false
image:
  caption: ''
  focal_point: ""
  placement: 2
  preview_only: false
lastmod: "2020-01-26T00:00:00Z"
projects: []
subtitle: 'Practical flow of a Data Science Project'
summary: Practical flow in conducting a Data Science Project in real business context'
tags:
- Process
- Business
title: 'Practical flow of a Data Science Project'
---

After 5 years of working in field of Analytics and Data Science in 2 multinational companies, the things that I learned through several failures and achievements are that the real bottleneck of a Data Scientist is the understanding of the data scientist's consumer rather than the technical side of running the right method. 

I will try to make this post condense in key points because practical application can vary in various forms and patterns, and it highly depends on the situation of a Data scientist's work. 

### 1. Clarify Business question

During my years in analytics and data science, in addition to technological concept explanation, **business question clarification** is one of the most difficult tasks when a Data scientist communicates with business partner. 

The reason is that the business partner always asks very general question such as *"I want to know why sales declined?"* or *"How to increase the efficiency of the promotion activity for brand A during summer?"* 

When you hear these questions, can you imagine the right approach or the answer to the case? I am sure that you will be very vague and confused, and if you deliver the result based on this current understanding, it is a high possibility that the business partners will say *"This is not what I need"*.

OMG! How terrible it is when you spent so much effort into this, but no one values it. 

> **This is because you did not touch the right pain point!**

I know that the common advice is to ask *"Why"* in order to dig into the real problems. However, this solution is not applicable all times because the business partners might not know why for all of your questions.

One thing that helped me overcome this difficulty is to **imagine the outcome that I desire**.

### 2. Identify the approach for the problem

This part is to set the methodology for the analysis.

Have you ever faced a situation where you do not know which approach to be used between descriptive analysis, predictive analysis, classification, segmentation, time series forecasting, etc.?

* Linear Regression cannot be used to segment customers or
* Descriptive analysis cannot predict customer churn. 

This step seems to be easy but in fact quickly got confused. If the Sales Director asked Data Science team to forecast the **sales for next year**, and the business need is to get the forecast **based on the amount of budget spending**, then which model should be used?
If the business wanted the forecast **based on the market movement**, which approach is suitable? 

This is the importance in choosing the right approach for the business question. 

### 3. Acquire the appropriate data

After identifying the business questions and the approach above, set up data requirement and extract the appropriate data from the data warehouse are the next thing.

Data selection sounds to be straightforward but indeed complicated. To solve the business questions, which kind of data is in need of. For example, will it need to have the birthday information of the customer if the task is to predict their churn probability?

Then, identify the approach to acquire the data. This can be from company's data warehouse, conducting survey collecting result, or government-owned data... 

**What occurs during Data Collection?**

One of the foreseen problems is the **unavailability of data**. 

This problem is very common in real case in which data is unable to capture at the moment of collection such as *the time spent in using a non-digital product*. As a common sense, you will think that you have to get the data. However, you have to consider the consequences of getting data including cost, time, resources and if the data is indeed not too important to your model, all the effort you put into it will be down the drain.

Therefore, the solution for this case is to defer inaccessible data and in case the model requires this data for a better result, you will have more confident to invest in obtaining it.

One of the provident ways of getting new features is to change the system of collecting data to acquire the right information needed. The Data scientist team can discuss with data management to get the timestamp that customers log on the page in addition to the browsing/ purchasing data. 

If the Data scientist is unable to acquire unavailable data through the above way, then think about contacting an outside data owner and prepare the budget.

After getting all the data you need, the next step is thing that a Data scientist usually does: 

* EDA
* Data cleaning
* Validation choosing
* Feature engineering
* Model selection/Building
* Feature selection
* Model tuning
* Ensemble

The order can be flexible and this is the standard progress that I usually do in my project and my job. Sometimes, after tuning and the accuracy does not meet my expectation, I need to go back to the feature engineering step to find other way to deal with features.