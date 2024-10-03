package com.unumunu.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme  // Material 3 MaterialTheme 임포트
import androidx.compose.material3.Surface      // Surface를 사용하려면 임포트 필요
import androidx.compose.runtime.Composable

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface {
                    AudioRecorder()
                }
            }
        }
    }
}
