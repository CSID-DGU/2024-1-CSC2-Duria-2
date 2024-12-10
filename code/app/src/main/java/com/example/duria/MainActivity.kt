package com.example.duria

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.example.duria.ui.ChatScreen
import com.example.duria.ui.theme.ChatBotAppTheme
import com.google.firebase.FirebaseApp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        FirebaseApp.initializeApp(this) // Firebase 초기화 (필요 시)
        setContent {
            ChatBotAppTheme {
                ChatScreen()
            }
        }
    }
}
