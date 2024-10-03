package com.unumunu.myapplication

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

object ServerService {
    private val client = OkHttpClient()

    suspend fun sendTextToServer(text: String): String? {
        return withContext(Dispatchers.IO) {
            val url = "http://YOUR_SERVER_IP:5000/chat"  // 실제 서버 주소로 변경

            val json = """{"instruction": "$text"}"""
            val requestBody = json.toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            try {
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    response.body?.string()
                } else {
                    "서버 오류: ${response.code}"
                }
            } catch (e: IOException) {
                e.printStackTrace()
                "네트워크 오류: ${e.message}"
            }
        }
    }
}
