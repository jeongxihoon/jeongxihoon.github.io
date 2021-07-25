---
title: "About Me"
layout: archieve
permalink: /about_me/
author_profile: true
sidebar_main: true
toc: true
toc_sticky: true
---
# Profile


- Name : Jihoon, Jeong


{% assign posts = site.categories.blog %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
