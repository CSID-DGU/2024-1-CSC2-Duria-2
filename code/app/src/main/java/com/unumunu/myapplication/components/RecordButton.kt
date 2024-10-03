package com.unumunu.myapplication.components

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.unumunu.myapplication.NaverSTTService
import com.unumunu.myapplication.services.ServerService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun RecordButton(onResponse: (String) -> Unit) {
    var isRecording by remember { mutableStateOf(false) }
    var isPlaying by remember { mutableStateOf(false) }
    var recordStatus by remember { mutableStateOf("대기 중") }
    var sttText by remember { mutableStateOf("") }  // STT 결과를 저장할 변수
    var serverResponseText by remember { mutableStateOf("") }  // 서버 응답을 저장할 변수
    var mediaPlayer: MediaPlayer? by remember { mutableStateOf(null) }
    var mediaRecorder: MediaRecorder? = null  // 초기에는 null로 설정
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var audioFilePath by remember { mutableStateOf("") }  // 녹음 파일 경로를 저장할 변수

    // 권한 요청 함수
    fun requestPermissions(): Boolean {
        val activity = context as? ComponentActivity
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)

        val neededPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
        }

        if (neededPermissions.isNotEmpty()) {
            activity?.let {
                ActivityCompat.requestPermissions(
                    it,
                    neededPermissions.toTypedArray(),
                    1
                )
            }
            return false
        }
        return true
    }

    // 녹음 시작 함수
    fun startRecording() {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir: File? = context.getExternalFilesDir(Environment.DIRECTORY_MUSIC)
        val audioFile = File.createTempFile("AUDIO_${timeStamp}_", ".mp3", storageDir)
        audioFilePath = audioFile.absolutePath  // 녹음 파일 경로 설정

        mediaRecorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setOutputFile(audioFilePath)
            try {
                prepare()
                start()
                recordStatus = "녹음 중..."
            } catch (e: IOException) {
                e.printStackTrace()
                recordStatus = "녹음 시작 실패"
            } catch (e: IllegalStateException) {
                e.printStackTrace()
                recordStatus = "녹음 시작 실패"
            }
        }
    }

    // 녹음 중지 함수
    fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
                recordStatus = "녹음 완료"
            }
            mediaRecorder = null
        } catch (e: IllegalStateException) {
            e.printStackTrace()
            recordStatus = "녹음 중지 실패"
            mediaRecorder = null
        }
    }

    // 녹음된 파일 재생 함수
    fun playAudio() {
        mediaPlayer = MediaPlayer().apply {
            try {
                setDataSource(audioFilePath)
                prepare()
                start()
                isPlaying = true
                setOnCompletionListener {
                    isPlaying = false
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    // 녹음된 파일 재생 중지 함수
    fun stopAudio() {
        mediaPlayer?.release()
        mediaPlayer = null
        isPlaying = false
    }

    // 녹음 버튼
    Button(onClick = {
        if (!isRecording) {
            // 녹음 시작
            if (requestPermissions()) {
                startRecording()
                isRecording = true
            }
        } else {
            // 녹음 중지 및 STT 및 서버 요청
            stopRecording()
            isRecording = false
            recordStatus = "처리 중..."

            coroutineScope.launch(Dispatchers.IO) {
                // STT API 호출하여 텍스트 반환
                val recognizedText = NaverSTTService.sendAudioToNaverSTT(audioFilePath)
                recognizedText?.let { text ->
                    sttText = text  // STT 결과를 화면에 출력

                    // 서버로 텍스트 전송 후 응답 받기
                    val serverResponse = ServerService.sendTextToServer(text)
                    serverResponse?.let { response ->
                        serverResponseText = response  // 서버 응답을 화면에 출력
                        onResponse(response)
                    } ?: run {
                        serverResponseText = "Error: 서버 응답 실패"
                        onResponse("Error: 서버 응답 실패")
                    }
                } ?: run {
                    sttText = "Error: STT 변환 실패"
                    onResponse("Error: STT 변환 실패")
                }
            }
        }
    }) {
        Text(text = if (isRecording) "녹음 중지" else "녹음 시작")
    }

    Spacer(modifier = Modifier.height(16.dp))

    // 재생 버튼
    Button(onClick = {
        if (isPlaying) {
            stopAudio()
        } else {
            playAudio()
        }
    }, enabled = !isRecording && audioFilePath.isNotEmpty()) {
        Text(text = if (isPlaying) "재생 중지" else "녹음 파일 재생")
    }

    Spacer(modifier = Modifier.height(16.dp))

    Text(text = "녹음 상태: $recordStatus")

    Spacer(modifier = Modifier.height(16.dp))

    // STT 결과 출력
    Text(text = "STT 결과: $sttText")

    Spacer(modifier = Modifier.height(16.dp))

    // 서버 응답 출력
    Text(text = "서버 응답: $serverResponseText")
}
