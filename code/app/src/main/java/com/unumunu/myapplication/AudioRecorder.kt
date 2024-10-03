package com.unumunu.myapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Environment
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button  // Material 3 Button 임포트
import androidx.compose.material3.Text    // Material 3 Text 임포트
import androidx.compose.material3.CircularProgressIndicator  // 로딩 인디케이터 추가
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment  // Alignment를 위해 추가
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun AudioRecorder() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var isRecording by remember { mutableStateOf(false) }
    var isPlaying by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }  // 로딩 상태 변수 추가
    var recorder: MediaRecorder? by remember { mutableStateOf(null) }
    var player: MediaPlayer? by remember { mutableStateOf(null) }
    var audioFilePath by remember { mutableStateOf("") }
    var sttResult by remember { mutableStateOf("") }
    var serverResponse by remember { mutableStateOf("") }  // 서버 응답 저장 변수

    // 권한 요청 런처
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            if (granted) {
                startRecording(context) { path, mediaRecorder ->
                    audioFilePath = path
                    recorder = mediaRecorder
                    isRecording = true
                }
            } else {
                // 권한 거부됨 처리
            }
        }
    )

    Column(
        modifier = Modifier.padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally  // 중앙 정렬
    ) {
        // 녹음 버튼
        Button(
            onClick = {
                if (isRecording) {
                    // 녹음 중지
                    stopRecording(recorder)
                    recorder = null
                    isRecording = false

                    // 로딩 시작
                    isLoading = true

                    // 녹음 중지 후 STT 요청
                    coroutineScope.launch {
                        sttResult = NaverSTTService.sendAudioToNaverSTT(audioFilePath)

                        // STT 결과가 있으면 서버로 전송
                        if (sttResult.isNotEmpty()) {
                            // 서버로 텍스트 전송
                            val response = ServerService.sendTextToServer(sttResult)
                            serverResponse = response ?: "서버 응답 오류"
                        } else {
                            serverResponse = "STT 변환 실패"
                        }

                        // 로딩 종료
                        isLoading = false
                    }
                } else {
                    // 녹음 시작
                    // 권한 체크
                    if (ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.RECORD_AUDIO
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        startRecording(context) { path, mediaRecorder ->
                            audioFilePath = path
                            recorder = mediaRecorder
                            isRecording = true
                            sttResult = ""  // 이전 STT 결과 초기화
                            serverResponse = ""  // 이전 서버 응답 초기화
                        }
                    } else {
                        // 권한 요청
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = if (isRecording) "녹음 중지" else "녹음 시작")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 재생 버튼
        Button(
            onClick = {
                if (isPlaying) {
                    stopPlaying(player)
                    player = null
                    isPlaying = false
                } else {
                    startPlaying(audioFilePath) { mediaPlayer ->
                        player = mediaPlayer
                        isPlaying = true
                        mediaPlayer.setOnCompletionListener {
                            isPlaying = false
                        }
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = audioFilePath.isNotEmpty() && !isRecording
        ) {
            Text(text = if (isPlaying) "재생 중지" else "녹음 파일 재생")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 로딩 인디케이터
        if (isLoading) {
            CircularProgressIndicator()
            Spacer(modifier = Modifier.height(16.dp))
        }

        // STT 결과 표시
        if (sttResult.isNotEmpty()) {
            Text(text = "STT 결과: $sttResult")
            Spacer(modifier = Modifier.height(16.dp))
        }

        // 서버 응답 표시
        if (serverResponse.isNotEmpty()) {
            Text(text = "서버 응답: $serverResponse")
        }
    }
}

// 나머지 함수들은 이전과 동일
fun startRecording(context: Context, onStarted: (String, MediaRecorder) -> Unit) {
    val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val fileName = "AUDIO_$timeStamp.m4a"
    val storageDir = context.getExternalFilesDir(Environment.DIRECTORY_MUSIC)
    val audioFile = File(storageDir, fileName)
    val audioFilePath = audioFile.absolutePath

    val recorder = MediaRecorder().apply {
        setAudioSource(MediaRecorder.AudioSource.MIC)
        setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        setOutputFile(audioFilePath)
        try {
            prepare()
            start()
            onStarted(audioFilePath, this)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}

fun stopRecording(recorder: MediaRecorder?) {
    recorder?.apply {
        try {
            stop()
            release()
        } catch (e: RuntimeException) {
            e.printStackTrace()
        }
    }
}

fun startPlaying(filePath: String, onStarted: (MediaPlayer) -> Unit) {
    val player = MediaPlayer().apply {
        try {
            setDataSource(filePath)
            prepare()
            start()
            onStarted(this)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}

fun stopPlaying(player: MediaPlayer?) {
    player?.apply {
        stop()
        release()
    }
}
