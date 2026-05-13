# 🦈 SharkSense – Global Shark Attack Risk Prediction using Environmental Data & Machine Learning

**SharkSense** is an end‑to‑end data analytics and AI project that predicts the probability of a fatal shark attack based on real‑time environmental conditions (sea surface temperature, air temperature, wind speed, moon phase). It uses only public datasets and free APIs – no hardware, no beach access required.

---

## 📌 Table of Contents

- [Idea & Motivation](#idea--motivation)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [What It Does (Step by Step)](#what-it-does-step-by-step)
- [Technology Stack](#technology-stack)
- [Results & Performance](#results--performance)
- [Future Integrations & Prospects](#future-integrations--prospects)
- [Ethical Considerations](#ethical-considerations)
- [Installation & Usage](#installation--usage)
- [License](#license)

---

## Idea & Motivation

Shark attacks are rare, but they create disproportionate fear and often lead to lethal control measures (shark nets, drumlines) that harm marine ecosystems. A non‑lethal, data‑driven alternative is to **forecast risk using environmental conditions** – the same approach used by marine biologists to study shark habitat.

**The core insight:**  
Shark behaviour and distribution are strongly influenced by water temperature, weather, and lunar cycles. By analysing historical attack records together with these environmental variables, we can train a machine learning model to recognise high‑risk conditions – and then apply that model to any beach in real time.

**Why this project is novel:**  
- Addresses the **geocoding problem** of most public attack datasets (coordinates often point to towns, not beaches).  
- Uses **free, no‑API‑key marine weather data** (Open‑Meteo).  
- Provides a **reproducible, open‑source pipeline** that can be extended globally.

---

## How It Works

The pipeline has three main phases:

1. **Coordinate Snapping** – Moves inland attack coordinates to the nearest coastline point (within 50 km) so that marine data can be retrieved.
2. **Environmental Enrichment** – For each incident, fetches:
   - Sea surface temperature (SST)
   - Air temperature (2 m)
   - Wind speed (10 m)
   - Moon illumination (calculated)
3. **Machine Learning** – Trains a Random Forest classifier to predict whether an incident was fatal (Y/N) based on those four features.

> **Note:** The model learns from past attacks. To forecast *future* risk, you simply fetch today’s environmental data for any beach and run the same model.

---

## System Architecture
