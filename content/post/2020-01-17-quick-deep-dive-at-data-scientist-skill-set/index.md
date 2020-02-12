---
authors:
- admin
categories:
- Data Science
date: "2020-01-17T00:00:00Z"
draft: false
featured: false
image:
  caption: ''
  focal_point: ""
  placement: 2
  preview_only: false
lastmod: "2020-01-28T00:00:00Z"
projects: []
subtitle: 'Skill set that Data Scientist needs to master'
summary: Skill set that Data Scientist needs to master through studying Kaggle survey 2019 using R'
tags:
- Skill
- Beginner
- Business
title: 'Quick deep dive at Data Scientist Skill set'
---

One year ago, when I truly and seriously considered improving my skill in Data Science, two questions were always lingering in my mind:

**What truly are the skills that a Data Scientist needs?**

**What skill will support me in the future and How do I improve myself in the right direction at the most efficacy?**

And many other friends, acquaintances reached out to me for a thought of what to learn to become a Data Scientist. 

Actually, I can share some of my experiences with them, but as you've already known, this field evolves unpreceedingly fast, technology and new required skills change on the yearly basis at the slowest.

> *What you learn today will be the old of tomorrow!*

Luckily, when I was known of the Kaggle dataset on Survey of Data Science and Machine learning, this data will definitely give me some insights for my questions.

The Data source is [here](https://www.kaggle.com/c/kaggle-survey-2019) 

In this post, I will summary my key findings from the survey. The complete analysis can be found through this [link](https://www.kaggle.com/geninhu/direction-toward-a-great-data-scientist/data)

*This analysis uses R language with tidyverse package for the best insight visualization.*

---

Before looking at the result, below are the data cleaning and preparation for the analysis.

1. Feature selection:

* This dataset includes the written answer of the respondents if they picked `"Other"` choice for some questions. However, the written answers are stored in another *csv* file so that all the variables containing `"Other"` will not a valuable variable for the analysis, therefore they will be excluded from the dataset.

* For this analysis, the `"Duration"` variable is not the factor that I want to explore so it will be excluded as well.

2. Other data cleaning: 

* Shorten some typical column names for easier understanding.

* Set level for values in the variables that require level later on, such as *Company size, Compensation/Salary, year of experience with machine learning*...

3. Create functions for plotting (Example is as below)

```python
plot_df_ds <- function (df_draw_ds, columnName1, columnName2) {
    names(df_draw_ds)[columnName1] <- paste("value")
    names(df_draw_ds)[columnName2] <- paste("value2")
    df_draw_ds <- df_draw_ds %>% 
    select (value, value2) %>%
    group_by(value, value2) %>%
    filter(value != "Not employed",value != "Other") %>% 
    summarise(count=n()) %>% 
    mutate(perc= prop.table(count))
```

---

Now, let's explore the result.

### **Role and Responsibilities:**

A Data scientist' is 's responsibility for many tasks with top 3 are:

* **Analyzing data** to influence product or to support business decisions

* **Explore new areas of Machine learning** through builidng prototypes

* Experiment to **improve company existing Machine learning models**

![](https://raw.githubusercontent.com/geniusnhu/ds-masterpiece-hugo/master/content/post/2020-01-17-quick-deep-dive-at-data-scientist-skill-set/responsibility.png)

### **Programming skills:**

**Python, SQL and R** are all the programming languages that are essential for Data Scientist as they stand in the Top 3 respectively.

<img src="https://raw.githubusercontent.com/geniusnhu/ds-masterpiece-hugo/master/content/post/2020-01-17-quick-deep-dive-at-data-scientist-skill-set/PL.png" width="300" height="450" >

### **Machine Learning skills:**

It is not a surprise if all Data Scientists use Machine Learning in their daily job but I was amazed that almost all of them use both Natural Language Processing and Computer Vision. These 2 fields has no longer been specified areas to some groups of users but expanded to much wider application and require Data Scientist to own these skills.

### **Machine Learning algorithm skills:**

**Regression** and **Tree-based algorithms** are the models used by Data Scientist. With the long history in statistics and analytics, these models are still being favor in analyzing data. Another advantage of these traditional models is that they are easy to explain to business partner. Other models such as Neuron network, or Deep learning is hard to explain the result. 

![](https://raw.githubusercontent.com/geniusnhu/ds-masterpiece-hugo/master/content/post/2020-01-17-quick-deep-dive-at-data-scientist-skill-set/ML.png)

Another interesting thing is that 51% of Data scientists is applying **Boosting method**.

### **Python skills:**

**Scikit-learn, Keras, Xgboost, TensorFlow and RandomForest libraries** are the top used frameworks given their benefit and convenience.

<img src="https://raw.githubusercontent.com/geniusnhu/ds-masterpiece-hugo/master/content/post/2020-01-17-quick-deep-dive-at-data-scientist-skill-set/package.png" width="300" height="450">

### **Data Scientist Credential:**

**Foundation skills**

* Writing code is an important skill of a good Data scientist. You do not need to be an excellent coder to be a Data scientist but it is dangerous to think that knowing little of coding skill is just enough to work as a Data scientist.

* It is essential to build up your business skills in addition to technical skills. Many data scientists, especially the junior excels at machine learning and coding but is lack of a business point of view so that most of their work does not support the business as the way it should be. Their recommendations were far from what the company is doing and it ended up beating all of their effort. That is the worst thing you want to face in the world of Data science when no one thinks your idea valuable.

**Specialized skills**

* Familiar with and expert in Python, SQL and/or R on various online/offline platforms.

* Fluency in using Local development environments and Intergrated development Environment, including but not limited to Jupyter, RStudio, Pycharm. Try to **learn on job** or **learn on practice**.

* Strong foundation with practical experience in Linear/Logistics regression, Decision Tree/Random Forest, and Gradient Boosting models. Moreover, besides popular and must known libraries such as Scikit-learn, Keras, Tensorflow, Xgboost is the framework that student should learn at the moment. Working on your own project/cooperating project is a good choice to get practical experience if you have not obtained the chance in a company. 

* Computer Vision and NLP are the booming areas in Data science so it is beneficial to prepare yourself with these skills.

* Although AutoML is a newly emerging field, it will probably become one of the important tools and skills for Data Scientist, including Automated hyperparameter tuning, Data augmentation, Feature engineering/selection and Auto ML pipelines.

* Skills in working with Big data and Big data products e.g Google Bigquerry, Databricks, Redshift are the must to own.

* This is the same with usage of cloud computing skills. In big company, it is a common to work on cloud so if you do not know how to conduct machine learning on cloud, it will become your minus comparing to other candidates.

And one last important thing: **Always believe in yourself, your choice and never give up**

#### End notes

As the big survey focusing on the Data science / Machine Learning areas, it appeared as a great source for me to gain valuable information. 

Beside understanding which typical skills requiring to advance in the Data Science field, I want to turn the dataset into a Salary prediction model and I will update on that in upcoming days.