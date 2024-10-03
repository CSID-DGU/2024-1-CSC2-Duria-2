package com.unumunu.myapplication.services

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

object ServerService {

    private val client = OkHttpClient()
    private const val TAG = "ServerService"

    // suspend 함수를 사용하여 서버로 텍스트를 전송하고 응답을 받는 함수
    suspend fun sendTextToServer(text: String): String? = withContext(Dispatchers.IO) {
        val url = "http://10.0.2.2:5000/chat"  // 실제 서버 URL과 엔드포인트로 변경
//        val url = "http://192.168.1.100:5000/chat"


        // JSON 형태로 요청 본문 생성
        val jsonMediaType = "application/json; charset=utf-8".toMediaTypeOrNull()
        val jsonBody = "{\"instruction\": \"$text\"}"
        val requestBody = jsonBody.toRequestBody(jsonMediaType)

        // 요청 빌드
        val request = Request.Builder()
            .url(url)
            .post(requestBody)
            .build()

        try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val responseText = response.body?.string()
                Log.d(TAG, "서버 응답 성공: $responseText")
                return@withContext responseText
            } else {
                Log.e(TAG, "서버 응답 실패: ${response.code} - ${response.message}")
                return@withContext null
            }
        } catch (e: IOException) {
            Log.e(TAG, "서버 요청 실패: ${e.message}")
            return@withContext null
        }
    }
}
