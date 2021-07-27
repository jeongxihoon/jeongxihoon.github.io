---
title: "Python"
layout: archive
permalink: categories/python
author_profile: true
sidebar_main: true
---
생초보의 Python 발전기


{% assign posts = site.categories.Python %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
