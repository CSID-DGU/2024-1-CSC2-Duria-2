package com.example.duria.services

import android.os.Build
import android.util.Log
import com.example.duria.models.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

object ServerService {
    // 타임아웃을 60초로 설정한 OkHttpClient 인스턴스 생성
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS) // 연결 타임아웃 60초
        .readTimeout(60, TimeUnit.SECONDS)    // 읽기 타임아웃 60초
        .writeTimeout(60, TimeUnit.SECONDS)   // 쓰기 타임아웃 60초
        .build()

    private const val TAG = "ServerService"

    suspend fun sendTextToServer(text: String): Message {
        return withContext(Dispatchers.IO) {
            val url = getServerUrl()

            val json = JSONObject().apply {
                put("user_input", text)
            }.toString()

            val requestBody = json.toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .addHeader("Content-Type", "application/json; charset=utf-8")
                .build()

            try {
                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonObject = JSONObject(responseBody)
                    val botResponse = jsonObject.getString("response")

                    // dialogue_state
                    val dialogueState = jsonObject.getJSONObject("dialogue_state")
                    val intent = dialogueState.getString("intent")
                    val emotion = dialogueState.getString("emotion")
                    val proactivity = dialogueState.optString("proactivity", null)
                    val consistency = dialogueState.optBoolean("consistency", true)

                    // slots 정보 추출
                    val slotsJson = dialogueState.getJSONObject("slots")
                    val slotMap = mutableMapOf<String, String>()

                    val keys = slotsJson.keys()
                    while (keys.hasNext()) {
                        val key = keys.next() as String
                        val value = slotsJson.getString(key)
                        slotMap[key] = value
                    }

                    // temp_slots 정보 추출
                    val tempSlotsJson = dialogueState.getJSONObject("temp_slots")
                    val tempSlotMap = mutableMapOf<String, String>()

                    val tempKeys = tempSlotsJson.keys()
                    while (tempKeys.hasNext()) {
                        val key = tempKeys.next() as String
                        val value = tempSlotsJson.getString(key)
                        tempSlotMap[key] = value
                    }

                    // Message 객체 생성
                    Message(
                        sender = "bot",
                        text = botResponse,
                        intent = intent,
                        emotion = emotion,
                        proactivity = proactivity,
                        consistency = consistency,
                        slots = slotMap,
                        temp_slots = tempSlotMap  // temp_slots 추가
                    )
                } else {
                    Log.e(TAG, "서버 오류: ${response.code} - ${response.message}")
                    Message(sender = "bot", text = "서버 오류: ${response.code}")
                }
            } catch (e: Exception) {
                e.printStackTrace()
                Log.e(TAG, "네트워크 오류: ${e.message}")
                Message(sender = "bot", text = "네트워크 오류: ${e.message}")
            }
        }
    }

    private fun getServerUrl(): String {
        return if (isEmulator()) {
            "http 주소"
        } else {
            "http 주소"
        }
    }

    private fun isEmulator(): Boolean {
        return (Build.FINGERPRINT.startsWith("generic")
                || Build.FINGERPRINT.lowercase().contains("virtual")
                || Build.MODEL.contains("Emulator")
                || Build.MODEL.contains("Android SDK built for x86"))
    }
}
