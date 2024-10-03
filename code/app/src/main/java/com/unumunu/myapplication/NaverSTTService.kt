package com.unumunu.myapplication

import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException

object NaverSTTService {
    private val client = OkHttpClient()

    data class STTResponse(
        val text: String
    )

    suspend fun sendAudioToNaverSTT(audioFilePath: String): String {
        return withContext(Dispatchers.IO) {
            val url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
            val audioFile = File(audioFilePath)

            if (!audioFile.exists()) {
                return@withContext "오디오 파일이 존재하지 않습니다."
            }

            val requestBody = audioFile.asRequestBody("application/octet-stream".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(url)
                .addHeader("X-NCP-APIGW-API-KEY-ID", BuildConfig.NAVER_CLIENT_ID)
                .addHeader("X-NCP-APIGW-API-KEY", BuildConfig.NAVER_CLIENT_SECRET)
                .post(requestBody)
                .build()

            try {
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    if (responseBody != null) {
                        val moshi = Moshi.Builder()
                            .addLast(KotlinJsonAdapterFactory())
                            .build()
                        val adapter = moshi.adapter(STTResponse::class.java)
                        val sttResponse = adapter.fromJson(responseBody)
                        sttResponse?.text ?: "응답에 텍스트가 없습니다."
                    } else {
                        "응답 본문이 없습니다."
                    }
                } else {
                    "오류 발생: ${response.code}"
                }
            } catch (e: IOException) {
                e.printStackTrace()
                "네트워크 오류"
            }
        }
    }
}
