# CCG Sleep Monitor

A DIY sleep monitoring system using single-lead ECG and machine learning for sleep stage classification, built on Raspberry Pi Pico.

## Project Overview

This project implements the breakthrough research from USC (2024) that achieves clinical-grade sleep staging using only single-lead ECG data, matching the accuracy of traditional polysomnography. The system is supposed to provide an affordable, comfortable alternative to expensive sleep labs for monitoring sleep patterns at home.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ECG Electrodes  │───▶│ AD8232 Analog    │───▶│ Raspberry Pi    │
│ (Chest Strap)   │    │ Front-End        │    │ Pico            │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐              │
                       │ MicroSD Card    │◀─────────────┘
                       │ Data Storage    │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ Python Analysis │
                       │ Sleep Staging   │
                       └─────────────────┘
```
