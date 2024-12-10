package com.example.duria.models

data class Message(
    val sender: String = "",
    val text: String = "",
    val timestamp: Long = System.currentTimeMillis(),
    val intent: String? = null,
    val emotion: String? = null,
    val proactivity: String? = null,
    val consistency: Boolean? = null,
    val slots: Map<String, String>? = null,
    val temp_slots: Map<String, String>? = null  // temp_slots 추가
)
