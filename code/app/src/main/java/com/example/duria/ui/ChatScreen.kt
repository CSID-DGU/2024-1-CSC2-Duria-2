package com.example.duria.ui

import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.duria.components.VoiceRecorderButton
import com.example.duria.models.Message
import com.example.duria.services.FirebaseService
import java.util.*

@Composable
fun ChatScreen() {
    val context = LocalContext.current
    val messages = remember { mutableStateListOf<Message>() }

    var isTTSInitialized by remember { mutableStateOf(false) }
    var textToSpeech by remember { mutableStateOf<TextToSpeech?>(null) }

    DisposableEffect(Unit) {
        // TextToSpeech 초기화
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.KOREAN)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("TTS", "한국어 언어 데이터가 없거나 지원되지 않습니다.")
                    isTTSInitialized = false
                } else {
                    isTTSInitialized = true
                }
            } else {
                Log.e("TTS", "TTS 초기화 실패")
                isTTSInitialized = false
            }
        }

        onDispose {
            textToSpeech?.stop()
            textToSpeech?.shutdown()
            textToSpeech = null
        }
    }

    LaunchedEffect(Unit) {
        FirebaseService.observeMessages { message ->
            messages.add(message)
            if (message.sender == "bot" && isTTSInitialized) {
                textToSpeech?.speak(message.text, TextToSpeech.QUEUE_ADD, null, null)
            }
        }
    }

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .padding(8.dp),
        ) {
            items(messages) { message ->
                MessageItem(message = message)
            }
        }
        HorizontalDivider()
        Row(
            modifier = Modifier
                .padding(8.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            VoiceRecorderButton()
        }
    }
}
