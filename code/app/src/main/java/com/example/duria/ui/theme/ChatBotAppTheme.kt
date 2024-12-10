package com.example.duria.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun ChatBotAppTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        // 여기에서 색상, 타이포그래피 등을 설정할 수 있습니다.
        content = content
    )
}
