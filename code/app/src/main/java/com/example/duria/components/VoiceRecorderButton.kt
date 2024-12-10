package com.example.duria.components

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Environment
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import com.example.duria.models.Message
import com.example.duria.services.FirebaseService
import com.example.duria.services.NaverSTTService
import com.example.duria.services.ServerService
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun VoiceRecorderButton() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var isRecording by remember { mutableStateOf(false) }
    var recorder: MediaRecorder? by remember { mutableStateOf(null) }
    var audioFilePath by remember { mutableStateOf("") }
    var startTime by remember { mutableStateOf(0L) }

    // 권한 요청 런처
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            if (granted) {
                startRecording(context) { path, mediaRecorder, recordingStartTime ->
                    audioFilePath = path
                    recorder = mediaRecorder
                    startTime = recordingStartTime
                    isRecording = true
                }
            } else {
                // 권한 거부 처리 (예: Toast 메시지 표시)
                // Toast.makeText(context, "오디오 녹음 권한이 필요합니다.", Toast.LENGTH_SHORT).show()
            }
        }
    )

    Button(onClick = {
        if (isRecording) {
            // 녹음 중이면 녹음 중지
            stopRecording(recorder)
            isRecording = false

            val endTime = System.currentTimeMillis()
            val durationInMillis = endTime - startTime
            val durationInSeconds = durationInMillis / 1000.0

            // 녹음 시간 출력 또는 사용
            Log.d("VoiceRecorder", "녹음 시간: $durationInSeconds 초")

            // 오디오 처리 시작
            coroutineScope.launch {
                processAudio(context, audioFilePath, durationInSeconds)
            }
        } else {
            // 녹음 시작
            if (ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.RECORD_AUDIO
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                startRecording(context) { path, mediaRecorder, recordingStartTime ->
                    audioFilePath = path
                    recorder = mediaRecorder
                    startTime = recordingStartTime
                    isRecording = true
                }
            } else {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }) {
        Text(text = if (isRecording) "녹음 중지" else "음성 입력")
    }
}

private fun startRecording(context: Context, onStarted: (String, MediaRecorder, Long) -> Unit) {
    val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val fileName = "AUDIO_$timeStamp.m4a"
    val storageDir = context.getExternalFilesDir(Environment.DIRECTORY_MUSIC)
    val audioFile = File(storageDir, fileName)
    val audioFilePath = audioFile.absolutePath

    try {
        val recorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setOutputFile(audioFilePath)
            prepare()
            start()
        }
        val startTime = System.currentTimeMillis()  // 녹음 시작 시간 기록
        onStarted(audioFilePath, recorder, startTime)
    } catch (e: Exception) {
        e.printStackTrace()
        // 에러 메시지를 사용자에게 표시하거나 처리 (예: Toast 메시지)
        // Toast.makeText(context, "녹음 시작 실패", Toast.LENGTH_SHORT).show()
    }
}

private fun stopRecording(recorder: MediaRecorder?) {
    recorder?.apply {
        try {
            stop()
            release()
        } catch (e: IllegalStateException) {
            e.printStackTrace()
        } catch (e: RuntimeException) {
            e.printStackTrace()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}

private suspend fun processAudio(
    context: Context,
    audioFilePath: String,
    durationInSeconds: Double
) {
    // 녹음 시간 사용
    Log.d("VoiceRecorder", "녹음 시간: $durationInSeconds 초")

    try {
        // STT 변환
        val recognizedText = NaverSTTService.sendAudioToNaverSTT(context, audioFilePath)

        // 사용자 메시지 생성
        val userMessage = Message(sender = "user", text = recognizedText)

        // 서버에 전송 및 응답 수신
        val botMessage = ServerService.sendTextToServer(recognizedText)

        // Firebase에 메시지 저장
        FirebaseService.sendMessage(userMessage)
        FirebaseService.sendMessage(botMessage)
    } catch (e: Exception) {
        Log.e("VoiceRecorder", "Error processing audio: ${e.message}")
        // 필요 시 사용자에게 에러 메시지 표시
    }
}
