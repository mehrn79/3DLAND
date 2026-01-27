document.addEventListener('DOMContentLoaded', () => {
    
    // 1. داده‌های نمودار (طبق تصویر Property III)
    const chartData = [
        { organ: "Liver", ours: 2929, other: 131, otherName: "LiTS", headerClass: "header-liver" },
        { organ: "Pancreas", ours: 1168, other: 420, otherName: "MSD Task 07", headerClass: "header-pancreas" },
        { organ: "Stomach", ours: 348, other: 300, otherName: "Stomach Cancer CT", headerClass: "header-stomach" },
        { organ: "Kidney", ours: 1217, other: 300, otherName: "KiTS", headerClass: "header-kidney" },
        { organ: "Spleen", ours: 358, other: 206, otherName: "RSNA RATIC", headerClass: "header-spleen" },
        { organ: "Gallbladder", ours: 30, other: 0, otherName: "No dataset", headerClass: "header-gallbladder" }
    ];

    const container = document.getElementById('chart-grid');

    if (container) {
        // 2. ساخت HTML کارت‌ها به صورت داینامیک
        chartData.forEach((item, index) => {
            // محاسبه مقیاس: بزرگترین عدد + ۲۰٪ فضای خالی
            const maxValue = Math.max(item.ours, item.other) * 1.2;
            
            const oursPercent = (item.ours / maxValue) * 100;
            const otherPercent = item.other > 0 ? (item.other / maxValue) * 100 : 0;

            const card = document.createElement('div');
            card.className = 'chart-card';
            // تاخیر آبشاری برای ظاهر شدن کارت‌ها
            card.style.transitionDelay = `${index * 100}ms`;

            // ساخت نوار "Other"
            let otherBarHTML = '';
            if (item.other > 0) {
                otherBarHTML = `
                    <div class="bar other" style="--target-width: ${otherPercent}%"></div>
                    <div class="bar-label">${item.other} <span>(${item.otherName})</span></div>
                `;
            } else {
                otherBarHTML = `<div class="no-dataset">No dataset</div>`;
            }

            card.innerHTML = `
                <div class="chart-header ${item.headerClass}">${item.organ}</div>
                
                <div class="bar-row">
                    <div class="bar ours" style="--target-width: ${oursPercent}%"></div>
                    <div class="bar-label">${item.ours} <span>(OURS)</span></div>
                </div>

                <div class="bar-row">
                    ${otherBarHTML}
                </div>
            `;

            container.appendChild(card);
        });

        // 3. انیمیشن با Intersection Observer (وقتی اسکرول رسید اجرا شود)
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const card = entry.target;
                    
                    // ظاهر شدن خود کارت
                    card.classList.add('visible');

                    // شروع پر شدن نوارها (با کمی تاخیر)
                    setTimeout(() => {
                        card.classList.add('active');
                        const bars = card.querySelectorAll('.bar');
                        bars.forEach(bar => {
                            const width = bar.style.getPropertyValue('--target-width');
                            bar.style.width = width;
                        });
                    }, 300);

                    observer.unobserve(card); // فقط یکبار اجرا شود
                }
            });
        }, { threshold: 0.15 });

        document.querySelectorAll('.chart-card').forEach(card => {
            observer.observe(card);
        });
    }
});