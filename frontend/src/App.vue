<template>
  <div class="app-container">
    <app-header />
    
    <el-main class="main-content">
      <el-card class="qa-card">
        <template #header>
          <div class="card-header">
            <span class="card-title">Ask a Question</span>
          </div>
        </template>
        
        <query-input v-model:question="question" />
        <method-selector v-model:method="retrievalMethod" />
        
        <div class="search-actions">
          <el-button 
            type="primary" 
            :loading="loading"
            @click="submitQuestion"
            class="search-button"
          >
            <el-icon><Search /></el-icon>
            <span>Search</span>
          </el-button>
        </div>
        
        <transition name="fade">
          <div v-if="answer || retrievedDocuments.length" class="result-container">
            <el-divider content-position="center">Results</el-divider>
            <answer-display :answer="answer" />
            <document-list :documents="retrievedDocuments" />
          </div>
        </transition>
      </el-card>
    </el-main>
    
    <app-footer />
  </div>
</template>

<script>
import { Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import AppHeader from './components/layout/AppHeader.vue'
import AppFooter from './components/layout/AppFooter.vue'
import QueryInput from './components/results/QueryInput.vue'
import MethodSelector from './components/results/MethodSelector.vue'
import AnswerDisplay from './components/results/AnswerDisplay.vue'
import DocumentList from './components/results/DocumentList.vue'

export default {
  name: 'App',
  components: {
    Search,
    AppHeader,
    AppFooter,
    QueryInput,
    MethodSelector,
    AnswerDisplay,
    DocumentList
  },
  data() {
    return {
      question: '',
      retrievalMethod: 'keyword',
      loading: false,
      answer: '',
      retrievedDocuments: []
    }
  },
  methods: {
    async submitQuestion() {
      if (!this.question.trim()) {
        ElMessage({
          message: 'Please enter a question',
          type: 'warning'
        })
        return
      }
      
      this.loading = true
      try {
        const response = await fetch('http://localhost:8000/api/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            question: this.question,
            retrieval_method: this.retrievalMethod
          })
        })
        
        const data = await response.json()
        
        // 使用动画效果显示结果
        setTimeout(() => {
          this.answer = data.answer
          this.retrievedDocuments = data.documents
          
          ElMessage({
            message: 'Query processed successfully',
            type: 'success'
          })
          this.loading = false
        }, 300)
      } catch (error) {
        console.error('Error:', error)
        ElMessage({
          message: 'An error occurred while processing your query',
          type: 'error'
        })
        this.loading = false
      }
    }
  }
}
</script>

<style>
@import './styles/global.css';

.main-content {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 40px 20px;
}

.qa-card {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--primary);
}

.search-actions {
  display: flex;
  justify-content: center;
  margin: 24px 0;
}

.search-button {
  padding: 12px 32px;
  font-size: 16px;
  gap: 8px;
}

.result-container {
  margin-top: 30px;
}
</style>