import { Link, useNavigate } from 'react-router-dom'

import { Card, MetricCard, MetricGrid, SectionHeading } from '../components/ui'
import { useWorkspace } from '../hooks/useWorkspace'

const STEPS = [
  {
    title: 'Upload a dataset',
    body: 'Bring a CSV or Excel file of student records. It is checked for missing values, duplicates and a usable target column as soon as it lands.',
    to: '/upload',
  },
  {
    title: 'Explore the data',
    body: 'Distributions, correlations, missing values and scatter comparisons, with filters you can try without touching the data used for training.',
    to: '/analysis',
  },
  {
    title: 'Train a model',
    body: 'Pick what to predict. The problem type is detected for you, a model is recommended, and the dataset is checked before any training starts.',
    to: '/train',
  },
  {
    title: 'Score students',
    body: 'Predict one student from a form built out of the model’s own features, or upload a file and score every row at once.',
    to: '/predict',
  },
  {
    title: 'Export the results',
    body: 'Download predictions as CSV, with confidence and predicted class included whenever the model can produce them.',
    to: '/predict',
  },
]

export function HomePage() {
  const { dataset, model } = useWorkspace()
  const navigate = useNavigate()

  return (
    <div className="space-y-8">
      <Card className="p-8 lg:p-10 border-brand/20 bg-gradient-to-br from-brand-soft/70 via-surface to-surface">
        <p className="text-[13px] font-semibold text-brand">Student performance prediction</p>
        <h1 className="font-display text-[34px] lg:text-[42px] leading-[1.08] font-semibold tracking-[-0.03em] text-ink mt-2 max-w-[20ch]">
          From a spreadsheet of student records to a working prediction.
        </h1>
        <p className="text-[15px] text-muted mt-4 max-w-[62ch] leading-relaxed">
          EduPredict AI walks a dataset through analysis, model training and prediction in one place.
          Nothing is sent anywhere: files are parsed, trained and scored by the backend running on
          this machine.
        </p>

        <div className="flex flex-wrap gap-3 mt-7">
          <button type="button" className="btn-primary" onClick={() => navigate('/upload')}>
            {dataset ? 'Upload a different dataset' : 'Upload your dataset'}
          </button>
          <Link to="/about" className="btn-secondary">
            Read about this project
          </Link>
          {dataset && (
            <Link to="/analysis" className="btn-ghost">
              Continue with {dataset.filename}
            </Link>
          )}
        </div>
      </Card>

      <div>
        <SectionHeading
          title="Where things stand"
          description="The workspace keeps your dataset and model between pages, so nothing is recomputed when you move around."
        />
        <MetricGrid columns={4}>
          <MetricCard
            label="Active dataset"
            value={dataset ? dataset.filename : 'None'}
            hint={dataset ? `${dataset.num_cols} columns` : 'Upload one to begin'}
            tone={dataset ? 'brand' : 'default'}
          />
          <MetricCard
            label="Rows loaded"
            value={dataset ? dataset.num_rows.toLocaleString() : '0'}
            hint={dataset ? `${dataset.total_missing.toLocaleString()} missing values` : 'Waiting for a file'}
          />
          <MetricCard
            label="Trained model"
            value={model ? model.model_choice : 'None'}
            hint={model ? `Predicting ${model.target_column}` : 'Train one after analysis'}
            tone={model ? 'brand' : 'default'}
          />
          <MetricCard
            label="Models available"
            value="6"
            hint="Three for classification, three for regression"
          />
        </MetricGrid>
      </div>

      <div>
        <SectionHeading title="How it works" description="Five steps, in order." />
        <ol className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {STEPS.map((step, index) => (
            <li key={step.title}>
              <Link
                to={step.to}
                className="card p-5 h-full block transition-shadow hover:shadow-lift focus-visible:shadow-lift"
              >
                <span className="w-6 h-6 rounded-full bg-brand-soft text-brand-deep grid place-items-center text-[12px] font-bold">
                  {index + 1}
                </span>
                <h3 className="font-display text-[15px] font-bold text-ink mt-3">{step.title}</h3>
                <p className="text-[13px] text-muted mt-1.5 leading-relaxed">{step.body}</p>
              </Link>
            </li>
          ))}
        </ol>
      </div>

      <Card>
        <SectionHeading
          title="What the model can predict"
          description="Target columns are detected by name, so the tool stays focused on student performance rather than becoming a general-purpose model builder."
        />
        <div className="flex flex-wrap gap-2">
          {[
            'exam_score',
            'final_score',
            'test_score',
            'final_grade',
            'placement_status',
            'performance_index',
            'pass_fail',
            'grade',
            'gpa',
            'cgpa',
            'result',
            'outcome',
          ].map((name) => (
            <code key={name} className="px-2.5 h-7 inline-flex items-center rounded-md bg-canvas border border-line text-[12px] text-ink font-mono">
              {name}
            </code>
          ))}
        </div>
        <p className="text-[13px] text-muted mt-4">
          If none of these appear in your file, you can still choose any column manually on the
          training page.
        </p>
      </Card>
    </div>
  )
}
