package com.example.duria.services

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import android.content.Context
import com.example.duria.BuildConfig
import okhttp3.MediaType.Companion.toMediaTypeOrNull


object NaverSTTService {
    private val client = OkHttpClient()

    suspend fun sendAudioToNaverSTT(context: Context, audioFilePath: String): String {
        return withContext(Dispatchers.IO) {
            val url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
            val audioFile = File(audioFilePath)

            if (!audioFile.exists()) {
                return@withContext "오디오 파일이 존재하지 않습니다."
            }

            val clientId = BuildConfig.NAVER_CLIENT_ID
            val clientSecret = BuildConfig.NAVER_CLIENT_SECRET

            val requestBody = audioFile.asRequestBody("application/octet-stream".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(url)
                .addHeader("X-NCP-APIGW-API-KEY-ID", clientId)
                .addHeader("X-NCP-APIGW-API-KEY", clientSecret)
                .post(requestBody)
                .build()

            try {
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    val jsonObject = JSONObject(responseBody)
                    jsonObject.getString("text")
                } else {
                    "STT 변환 오류: ${response.code}"
                }
            } catch (e: Exception) {
                e.printStackTrace()
                "네트워크 오류: ${e.message}"
            }
        }
    }
}
